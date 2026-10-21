// Exact per-step schedules of the current low, medium, and high thinking presets.
(async () => {
  const response = await fetch("data/recycling-schedules.json", {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Recycling schedules unavailable");
  const data = await response.json();
  const root = document.querySelector("#recycling-plots");
  const maxPasses = Math.max(...data.profiles.flatMap((profile) => profile.passes));
  for (const profile of data.profiles) {
    const card = document.createElement("div");
    card.className = "recycling-profile";
    card.style.setProperty(
      "--schedule-color",
      `var(--${profile.color === "teal" ? "green" : profile.color})`,
    );
    const x = (step) => 32 + (step / profile.steps) * 270;
    const y = (passes) => 143 - (passes / maxPasses) * 126;
    let path = `M${x(0)} ${y(profile.passes[0])}`;
    for (let i = 0; i < profile.steps; i++) {
      path += `H${x(i + 1)}`;
      if (i + 1 < profile.steps) path += `V${y(profile.passes[i + 1])}`;
    }
    const description = profile.stages
      .map(([length, count], i) => {
        const first =
          profile.stages.slice(0, i).reduce((sum, stage) => sum + stage[0], 0) +
          1;
        return `Steps ${first}–${first + length - 1}: ${count} ${count === 1 ? "trunk evaluation" : "trunk evaluations"}.`;
      })
      .join(" ");
    card.innerHTML = `<div class="schedule-heading"><h4>${profile.label}</h4><p class="schedule-use-case">${profile.total_passes} trunk evaluations</p></div>
      <svg viewBox="0 0 320 183" tabindex="0" role="img" aria-label="${profile.label}. Trunk evaluations per denoising step. ${description}">
        <title>${profile.label}: trunk evaluations per denoising step</title>
        ${[1, 3, 6, 9].map((value) => `<line x1="32" x2="302" y1="${y(value)}" y2="${y(value)}" class="schedule-grid"/><text x="22" y="${y(value) + 4}" text-anchor="end">${value}</text>`).join("")}
        <path d="${path}L${x(profile.steps)} 143H32Z" class="schedule-area"/>
        <path d="${path}" class="schedule-line"/>
        <text x="32" y="166">Early</text><text x="302" y="166" text-anchor="end">Late</text>
        <line class="schedule-guide" x1="0" x2="0" y1="12" y2="143" visibility="hidden"/>
        <circle class="schedule-marker" r="4" visibility="hidden"/>
      </svg><p class="schedule-readout" aria-live="polite"></p>`;
    const svg = card.querySelector("svg"),
      guide = card.querySelector(".schedule-guide"),
      marker = card.querySelector(".schedule-marker"),
      readout = card.querySelector(".schedule-readout");
    let activeStep = 29;
    const inspect = (step) => {
      activeStep = Math.max(0, Math.min(profile.steps - 1, step));
      const px = x(activeStep + 0.5),
        value = profile.passes[activeStep];
      guide.setAttribute("x1", px);
      guide.setAttribute("x2", px);
      guide.setAttribute("visibility", "visible");
      marker.setAttribute("cx", px);
      marker.setAttribute("cy", y(value));
      marker.setAttribute("visibility", "visible");
      readout.textContent = `Step ${activeStep + 1} · ${value} ${value === 1 ? "evaluation" : "evaluations"}`;
    };
    svg.addEventListener("pointermove", (event) => {
      const bounds = svg.getBoundingClientRect();
      inspect(
        Math.floor(
          ((((event.clientX - bounds.left) / bounds.width) * 320 - 32) / 270) *
            profile.steps,
        ),
      );
    });
    svg.addEventListener("pointerleave", () => {
      if (document.activeElement === svg) return;
      guide.setAttribute("visibility", "hidden");
      marker.setAttribute("visibility", "hidden");
      readout.textContent = "";
    });
    svg.addEventListener("focus", () => inspect(activeStep));
    svg.addEventListener("blur", () => {
      guide.setAttribute("visibility", "hidden");
      marker.setAttribute("visibility", "hidden");
      readout.textContent = "";
    });
    svg.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      inspect(activeStep + (event.key === "ArrowRight" ? 1 : -1));
    });
    root.append(card);
  }
})().catch((error) => {
  document.querySelector("#recycling-plots").textContent =
    "Recycling schedules could not load.";
  console.error(error);
});
