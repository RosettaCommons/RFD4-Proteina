import "./recycling.js?v=unified-reasoning";

const trajectoryVideo = document.querySelector("#full-trajectory-video");
const trajectoryPanel = document.querySelector("#reasoning-trajectory");
const trajectoryMotion = matchMedia("(prefers-reduced-motion: reduce)");
let trajectoryVisible = false;
function updateTrajectory() {
  if (
    trajectoryVisible &&
    !trajectoryPanel.hidden &&
    !document.hidden &&
    !trajectoryVideo.src
  ) {
    trajectoryVideo.src = trajectoryVideo.dataset.src;
  }
  if (
    trajectoryVisible &&
    !trajectoryPanel.hidden &&
    !document.hidden &&
    !trajectoryMotion.matches
  ) {
    trajectoryVideo.play().catch(() => {});
  } else trajectoryVideo.pause();
}
const trajectoryObserver = new IntersectionObserver(([entry]) => {
  trajectoryVisible = entry.isIntersecting;
  updateTrajectory();
});
trajectoryObserver.observe(trajectoryVideo);
document.addEventListener("visibilitychange", updateTrajectory);
trajectoryMotion.addEventListener("change", updateTrajectory);

const tabs = [...document.querySelectorAll(".reasoning-tabs [role=tab]")];
function selectReasoning(mode) {
  for (const tab of tabs) {
    const active = tab.dataset.reasoning === mode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
  }
  updateTrajectory();
}
for (const tab of tabs) {
  tab.addEventListener("click", () => selectReasoning(tab.dataset.reasoning));
  tab.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(tab);
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft")
      next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    selectReasoning(tabs[next].dataset.reasoning);
    tabs[next].focus();
  });
}
const requested = new URL(location.href).searchParams.get("reasoning");
selectReasoning(
  tabs.some((tab) => tab.dataset.reasoning === requested)
    ? requested
    : "shared",
);

// Reveal a tab panel before scrolling to a jump link or a bookmarked fragment.
function revealReasoningAnchor() {
  const panelId = location.hash.slice(1);
  const tab = tabs.find((tab) => tab.getAttribute("aria-controls") === panelId);
  if (!tab) return;
  selectReasoning(tab.dataset.reasoning);
  document.getElementById(panelId).scrollIntoView({ block: "start" });
}
window.addEventListener("hashchange", revealReasoningAnchor);
for (const link of document.querySelectorAll(".jump-card")) {
  link.addEventListener("click", () => {
    // Clicking the current fragment does not dispatch another hashchange.
    if (link.hash === location.hash) revealReasoningAnchor();
  });
}
revealReasoningAnchor();
