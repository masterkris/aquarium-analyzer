document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("myFile");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log("Analysis result:", data);
    } catch (error) {
      console.error("Upload failed:", error);
    }
  });
});
