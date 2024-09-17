document.addEventListener("DOMContentLoaded", function() {
  const langSelector = document.getElementById("language-selector");

  // Load the selected language from localStorage if available
  const savedLang = localStorage.getItem("selectedLanguage");
  if (savedLang) {
    langSelector.value = savedLang;
  }

  // Function to show/hide code examples based on the selected language
  function updateExamples() {
    const selectedLang = langSelector.value;

    // Save the selected language to localStorage
    localStorage.setItem("selectedLanguage", selectedLang);

    document.querySelectorAll(".example").forEach(function(example) {
      if (example.classList.contains(selectedLang)) {
        example.style.display = "block";
      } else {
        example.style.display = "none";
      }
    });
  }

  // Listen for changes in the dropdown menu
  langSelector.addEventListener("change", updateExamples);

  // Initialize by showing the appropriate examples when the page loads
  updateExamples();
});
