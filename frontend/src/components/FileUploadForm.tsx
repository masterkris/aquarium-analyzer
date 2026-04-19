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
      formData.append("tankSize", String(data.tankSize));

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
    <div className="max-w-md w-full mx-auto mt-16 p-8 rounded-2xl border border-(--border) shadow-md bg-(--bg)">
      <h1 className="text-2xl font-semibold text-center mb-6 text-(--text-h)">
        Aquarium Analyzer
      </h1>

      <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
        <div>
          <div className="flex flex-row items-center gap-2 mb-2">
            <label htmlFor="tankSize">Tank size (gallons):</label>
            <input
              className="text-sm text-(--text) cursor-pointer
              file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0
              file:text-sm file:font-semibold
              file:bg-(--accent-bg) file:text-(--accent)
              hover:file:opacity-80 file:cursor-pointer file:transition-opacity"
              type="number"
              id="tankSize"
              accept="image/*"
              {...register("tankSize")}
            />
          </div>
          <div className="flex flex-row items-center gap-2 mb-2">
            <input
              className="w-full text-sm text-(--text) cursor-pointer
              file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0
              file:text-sm file:font-semibold
              file:bg-(--accent-bg) file:text-(--accent)
              hover:file:opacity-80 file:cursor-pointer file:transition-opacity"
              type="file"
              id="myFile"
              accept="image/*"
              {...register("selectFile")}
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full py-2 px-4 rounded-lg font-semibold bg-(--accent) text-white
            hover:opacity-90 transition-opacity
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Analyzing..." : "Upload file"}
        </button>
      </form>

      {(loading || error || result) && (
        <section className="mt-6">
          {loading && (
            <p className="text-sm text-(--text) animate-pulse text-center">
              Analyzing your image...
            </p>
          )}
          {error && <p className="text-sm text-red-500">{error}</p>}
          {result && (
            <div className="p-4 rounded-xl border border-(--border) space-y-2">
              <h2 className="text-lg font-semibold text-(--text-h)">
                Total fish: {result.total_fish}
              </h2>
              <h3 className="text-sm font-medium text-(--text) uppercase tracking-wide mt-3">
                Species counts
              </h3>
              <ul className="text-sm text-(--text) space-y-1">
                {Object.entries(result.species_counts).map(
                  ([species, count]) => (
                    <li key={species}>
                      {species}: {count}
                    </li>
                  ),
                )}
              </ul>
              <h3 className="text-sm font-medium text-(--text) uppercase tracking-wide mt-3">
                Detections
              </h3>
              <ul className="text-sm text-(--text) space-y-1">
                {result.detections.map((det, i) => (
                  <li key={i}>
                    {det.species} — {det.confidence}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export { FileUploadForm };
