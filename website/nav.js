// Light/dark theme toggle and the BibTeX copy button.

function copyBibtex(id, btn) {
  var text = document.getElementById(id).textContent;
  var done = function () {
    btn.textContent = "Copied!";
    setTimeout(function () {
      btn.textContent = "Copy BibTeX";
    }, 2000);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(function () {
      fallbackCopy(text, done);
    });
  } else fallbackCopy(text, done);
}
function fallbackCopy(text, done) {
  var ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  document.body.removeChild(ta);
  done();
}
