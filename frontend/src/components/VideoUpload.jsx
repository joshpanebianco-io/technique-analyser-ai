import { useState } from "react";

export default function VideoUpload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0); // For progress bar
  const [analysisResult, setAnalysisResult] = useState(null); // To store score and suggestions
  const [showRepDetails, setShowRepDetails] = useState(false);

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file) {
      setStatus("❌ No file selected");
      return;
    }

    setStatus("⬆️ Uploading...");
    setUploadProgress(0); // Reset progress
    setAnalysisResult(null); // Clear previous results when starting a new upload

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Use XMLHttpRequest for progress tracking, as fetch API doesn't support it natively for uploads
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "http://localhost:8000/upload");

      // Event listener for upload progress
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
          const percentCompleted = Math.round((event.loaded * 100) / event.total);
          setUploadProgress(percentCompleted);

          // As soon as upload reaches 100%, assume analysis begins
          if (percentCompleted === 100) {
            setTimeout(() => {
              setStatus("🔍 Analysing...");
            }, 1500); // 2-second delay before changing status
          }
        }
      });

      // Event listener for when the upload is complete (or errors)
      xhr.onload = () => {
        if (xhr.status === 200) {
          setStatus("🔍 Analysing...");

          // Simulate slight wait for UI update (optional)
          setTimeout(() => {
            const data = JSON.parse(xhr.responseText);
            setStatus(`✅ Analysis complete: ${data.filename}`);
            setAnalysisResult(data);
          }, 100); // just enough to show the spinner
        } else {
          const errorData = JSON.parse(xhr.responseText);
          setStatus(`❌ Upload/Analysis failed: ${errorData.detail || "Unknown error"}`);
          setAnalysisResult(null);
        }
      };

      // Event listener for network errors
      xhr.onerror = () => {
        setStatus("❌ Network error or server unreachable");
        setAnalysisResult(null); // Clear results on network error
      };

      // Send the FormData
      xhr.send(formData);
    } catch (err) {
      console.error("Client-side error during upload:", err);
      setStatus("❌ Error uploading file");
      setAnalysisResult(null); // Clear results on client-side error
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-start px-6 pt-[15vh]">
      <div className="bg-white shadow-md rounded-xl p-6 w-full max-w-md">
        <h2 className="text-2xl font-semibold text-center mb-4 text-gray-800">
          Upload Exercise Video
        </h2>
        <form onSubmit={handleUpload} className="flex flex-col gap-4">
          <input
            type="file"
            accept="video/*"
            onChange={(e) => {
              setFile(e.target.files[0]);
              setStatus(""); // Clear status when a new file is selected
              setAnalysisResult(null); // Clear previous results
            }}
            className="file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0
                       file:text-sm file:font-semibold file:bg-blue-600 file:text-white
                       hover:file:bg-blue-700 cursor-pointer"
          />
          <button
            type="submit"
            disabled={!file} // Disable button if no file is selected
            className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700
                       transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Upload
          </button>
        </form>

        {status && <p className="mt-4 text-sm text-center text-gray-700">{status}</p>}

        {status.startsWith("🔍 Analysing") && (
          <div className="flex justify-center items-center mt-4">
            <div className="relative w-8 h-8">
              <div
                className="absolute w-full h-full border-4 border-blue-600 rounded-full animate-spin"
                style={{ borderTopColor: "transparent" }}
              ></div>
              <div
                className="absolute w-full h-full border-4 border-blue-300 rounded-full animate-spin"
                style={{ animationDuration: "1.5s", borderBottomColor: "transparent" }}
              ></div>
            </div>
          </div>
        )}

        {/* 
        {status.startsWith("🔍 Analysing") && (
          <div className="flex items-center justify-center mt-4 space-x-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
          </div>
        )} 
        */}

        {/* Progress Bar Display */}
        {status.startsWith("⬆️ Uploading...") && uploadProgress > 0 && (
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-4">
            <div
              className="bg-blue-500 h-2.5 rounded-full"
              style={{ width: `${uploadProgress}%` }}
            ></div>
            <p className="text-sm text-center text-gray-600 mt-1">{uploadProgress}%</p>
          </div>
        )}

        {/* Analysis Results Display */}
        {analysisResult && (
          <div className="mt-6 pt-4 border-t border-gray-200">
            <h3 className="text-xl font-semibold text-gray-800 mb-3">Analysis Report:</h3>
            <p className="text-lg text-gray-700 mb-2">
              <span className="font-bold">Score:</span> {analysisResult.avg_score}%
            </p>

            <div className="text-gray-700">
              <span className="font-bold">Suggestions for Improvement:</span>
              <ul className="list-disc list-inside mt-2 space-y-1 text-base">
                {analysisResult.suggestions.map((s, index) => (
                  <li key={index}>{s}</li>
                ))}
              </ul>

              {/* Collapsible Rep-by-rep breakdown */}
              {analysisResult.rep_feedback && analysisResult.rep_feedback.length > 0 && (
                <div className="mt-6">
                  <button
                    onClick={() => setShowRepDetails(!showRepDetails)}
                    className="font-bold text-blue-600 hover:underline focus:outline-none"
                    aria-expanded={showRepDetails}
                    aria-controls="rep-breakdown"
                  >
                    Rep-by-rep breakdown {showRepDetails ? "▲" : "▼"}
                  </button>

                  {showRepDetails && (
                    <ul
                      id="rep-breakdown"
                      className="list-decimal list-inside mt-2 space-y-1 text-base text-gray-700"
                    >
                      {analysisResult.rep_feedback.map((rep) => (
                        <ul key={rep.rep_number}>
                          <strong>Rep {rep.rep_number}:</strong> {rep.score}% — {rep.depth_feedback} |{" "}
                          {rep.posture_feedback}
                        </ul>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
