// Only the four co-first authors are shuffled; the rest of the byline stays in authors.tex order.
const byline = document.querySelector("#co-first-authors");
const authors = [...byline.querySelectorAll(".author-name")];
for (let i = authors.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [authors[i], authors[j]] = [authors[j], authors[i]];
}
byline.replaceChildren(
  ...authors.flatMap((author, index) =>
    index ? [document.createTextNode(", "), author] : [author],
  ),
);
