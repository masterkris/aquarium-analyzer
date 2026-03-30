document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("myFile");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const resultsDiv = document.getElementById("results");
    resultsDiv.textContent = "Analyzing...";

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      // After getting the response, display it on the page
      resultsDiv.innerHTML = "";
      resultsDiv.textContent = JSON.stringify(data);
    } catch (error) {
      resultsDiv.textContent = `Upload failed: ${error.message}`;
    }
  });
});
