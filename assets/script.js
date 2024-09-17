document.addEventListener("DOMContentLoaded", function() {
  const langSelector = document.getElementById("language-selector");

  function updateExamples() {
    const selectedLang = langSelector.value;
      document.querySelectorAll(".example").forEach(function(example) {
	  console.log(selectedLang);
      if (example.classList.contains(selectedLang)) {
        example.style.display = "block";
      } else {
        example.style.display = "none";
      }
    });
  }

  langSelector.addEventListener("change", updateExamples);
  updateExamples();  // Initialize on load
});
