document.addEventListener("DOMContentLoaded", () => {
  const beginBtn    = document.getElementById("begin-btn");
  const introPanel  = document.getElementById("intro-panel");
  const uploadCard  = document.getElementById("upload-card");

  if (!beginBtn || !introPanel || !uploadCard) return;

  beginBtn.addEventListener("click", () => {
    introPanel.classList.add("hide");  // hide the landing logo + button
    uploadCard.classList.add("show");  // reveal the upload box
    // optional: scroll to it smoothly
    uploadCard.scrollIntoView({behavior:"smooth", block:"start"});
  });
});
