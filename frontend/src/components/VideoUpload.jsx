import { useState } from "react";

export default function VideoUpload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file) {
      setStatus("❌ No file selected");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setStatus(`✅ Uploaded: ${data.filename}`);
      } else {
        setStatus("❌ Upload failed");
      }
    } catch (err) {
      console.error(err);
      setStatus("❌ Error uploading file");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-start px-6 pt-[15vh]">
      <div className="bg-white shadow-md rounded-xl p-6 w-full max-w-md">
        <h2 className="text-2xl font-semibold text-center mb-4">
          Upload Exercise Video
        </h2>
        <form onSubmit={handleUpload} className="flex flex-col gap-4">
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files[0])}
            className="file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
          />
          <button
            type="submit"
            className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition"
          >
            Upload
          </button>
        </form>
        {status && <p className="mt-4 text-sm text-center text-gray-700">{status}</p>}
      </div>
    </div>
  );
}
