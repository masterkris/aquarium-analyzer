import { useCallback, useState, type ReactElement } from "react";
import { useForm, type SubmitHandler } from "react-hook-form";
import axios from "axios";
import { type IFileUploadForm, type IAnalyzeResponse } from "../types";

function FileUploadForm(): ReactElement {
  const { register, handleSubmit } = useForm<IFileUploadForm>();
  const [result, setResult] = useState<IAnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit: SubmitHandler<IFileUploadForm> = useCallback(
    async (data: IFileUploadForm) => {
      const file = data.selectFile[0];
      if (!file) return;

      setLoading(true);
      setError(null);
      setResult(null);

      const formData = new FormData();
      formData.append("file", file);

      try {
        const { data } = await axios.post<IAnalyzeResponse>(
          "/analyze",
          formData,
        );
        setResult(data);
      } catch (err) {
        const message = axios.isAxiosError(err)
          ? (err.response?.statusText ?? err.message)
          : String(err);
        setError(`Upload failed: ${message}`);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return (
    <>
      <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
        <label htmlFor="myFile">Select a file:</label>
        <input
          className="border"
          type="file"
          id="myFile"
          accept="image/*"
          {...register("selectFile")}
        />
        <button type="submit" className="border">
          Upload file
        </button>
      </form>

      <section>
        {loading && <p>Analyzing...</p>}
        {error && <p>{error}</p>}
        {result && (
          <div>
            <h2>Total fish: {result.total_fish}</h2>
            <h3>Species counts</h3>
            <ul>
              {Object.entries(result.species_counts).map(([species, count]) => (
                <li key={species}>
                  {species}: {count}
                </li>
              ))}
            </ul>
            <h3>Detections</h3>
            <ul>
              {result.detections.map((det, i) => (
                <li key={i}>
                  {det.species} — {det.confidence}
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </>
  );
}

export { FileUploadForm };
