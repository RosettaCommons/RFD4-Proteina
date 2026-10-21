// Only the four co-first authors are shuffled; the rest of the byline stays in authors.tex order.
const byline = document.querySelector("#co-first-authors");
function shuffleAuthors(ensureChange = false) {
  const previous = [...byline.querySelectorAll(".author-name")];
  const authors = [...previous];
  do {
    for (let i = authors.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [authors[i], authors[j]] = [authors[j], authors[i]];
    }
  } while (ensureChange && authors.length > 1 && authors.every((author, i) => author === previous[i]));
  byline.replaceChildren(
    ...authors.flatMap((author, index) =>
      index ? [document.createTextNode(", "), author] : [author],
    ),
  );
}
shuffleAuthors();
byline.setAttribute("aria-live", "polite");
byline.setAttribute("aria-atomic", "true");
const shuffleButton = document.querySelector("#shuffle-authors");
shuffleButton.hidden = false;
shuffleButton.addEventListener("click", () => shuffleAuthors(true));
