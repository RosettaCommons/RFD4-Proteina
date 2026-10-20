/* Decoding of the explainer's figure data (docs/_static/explainer/data/<figure>.json).
 *
 * Bulky tensors are stored as quantised typed arrays:
 *
 *   {"dtype": "int8|uint8|int16|uint16|float32", "shape": [d0, d1, ...], "scale": s, "data": "<base64>"}
 *
 * `data` holds prod(shape) little-endian values in row-major order and the real value is
 * `int * scale`. `scale` is omitted for integer codes (residue classes, indices, masks), which
 * decode to the integers themselves. The exporters under docs/architecture/explainer/ write the
 * format (dxexport.packed) and tests/test_explainer_docs.py checks every field's byte length.
 *
 * No side effects: figures and tests import it directly.
 */

const TYPES = {
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  float32: Float32Array,
};

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** True when `field` is a quantised typed array rather than plain JSON. */
export const isPacked = (field) =>
  field !== null && typeof field === 'object' && 'dtype' in field && 'shape' in field && 'data' in field;

/** The raw little-endian integers (or float32 values) of a packed field, in a typed array. */
export function decodeRaw(field) {
  const Type = TYPES[field.dtype];
  if (!Type) throw new Error(`unknown dtype ${field.dtype}`);
  const text = atob(field.data);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  const count = field.shape.reduce((a, b) => a * b, 1);
  if (bytes.length !== count * Type.BYTES_PER_ELEMENT) {
    throw new Error(`${field.dtype}${JSON.stringify(field.shape)} needs ${count * Type.BYTES_PER_ELEMENT} bytes, got ${bytes.length}`);
  }
  if (LITTLE_ENDIAN || Type.BYTES_PER_ELEMENT === 1) return new Type(bytes.buffer);
  const view = new DataView(bytes.buffer);
  const out = new Type(count);
  const read = {
    int16: (o) => view.getInt16(o, true),
    uint16: (o) => view.getUint16(o, true),
    float32: (o) => view.getFloat32(o, true),
  }[field.dtype];
  for (let i = 0; i < count; i++) out[i] = read(i * Type.BYTES_PER_ELEMENT);
  return out;
}

/**
 * Decode a packed field.
 * @param {{dtype: string, shape: number[], scale?: number, data: string}} field
 * @returns {{values: Float32Array|Int8Array|Uint8Array|Int16Array|Uint16Array, shape: number[]}}
 *   `values` is a Float32Array of `int * scale` when the field has a scale, else the raw typed array.
 */
export function unpack(field) {
  const raw = decodeRaw(field);
  if (field.scale === undefined || field.scale === null) return { values: raw, shape: field.shape.slice() };
  const values = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) values[i] = raw[i] * field.scale;
  return { values, shape: field.shape.slice() };
}

/**
 * The block of an unpacked array at leading indices, as a view without copying:
 * `take(xyz, 3)` of a [60, N, 3] trajectory is frame 3 as a flat [N * 3] array.
 */
export function take({ values, shape }, ...indices) {
  let offset = 0;
  let stride = values.length;
  for (let axis = 0; axis < indices.length; axis++) {
    stride /= shape[axis];
    if (indices[axis] < 0 || indices[axis] >= shape[axis]) throw new RangeError(`index ${indices[axis]} on axis ${axis} of ${shape}`);
    offset += indices[axis] * stride;
  }
  return values.subarray(offset, offset + stride);
}
