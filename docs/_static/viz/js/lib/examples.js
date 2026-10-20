/* The example inputs of data/shared-examples.json, read the same way on every page that offers them.
 *
 *   exampleLength(draws, B)    the token length a batch of the first B recorded draws runs at: generate.py pads
 *                              a batch to the length bucket of its longest draw (draws.batch_padded_tokens, read
 *                              from its collation); a larger batch runs at draws.padded_possible_max, the bucket
 *                              of the longest design the input can yield
 *   tokenRoles(given, chain)   each token's role, as the attention and layer pages colour it: 'designed' when its
 *                              coordinates are not given, 'context' when they are and its whole chain is given (a
 *                              target), 'given' otherwise (a motif); the palette roles of the same names
 *
 * Importing this module has no side effects.
 */

/**
 * The token length a batch of B draws of an example runs at.
 * @param {object} draws  an example's `draws` in data/shared-examples.json
 * @param {number} B      the batch size
 */
export function exampleLength(draws, B) {
  const lengths = draws.batch_padded_tokens;
  return B <= lengths.length ? lengths[B - 1] : draws.padded_possible_max;
}

/**
 * Each token's role: 'designed', 'given' (a motif token) or 'context' (a given token of a wholly given chain).
 * @param {ArrayLike<number>} given  1 where the token's coordinates are given
 * @param {ArrayLike<number>} chain  each token's chain index
 * @returns {string[]}
 */
export function tokenRoles(given, chain) {
  const size = new Map();
  const fixed = new Map();
  Array.from(chain, (c, i) => {
    size.set(c, (size.get(c) || 0) + 1);
    fixed.set(c, (fixed.get(c) || 0) + (given[i] ? 1 : 0));
  });
  return Array.from(chain, (c, i) => (!given[i] ? 'designed' : fixed.get(c) === size.get(c) ? 'context' : 'given'));
}
