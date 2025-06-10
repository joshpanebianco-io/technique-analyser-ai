import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import VideoUpload from "./components/VideoUpload";

function App() {
  return (
    <div className="bg-gray-100">
      <header className="text-center py-6">
        <h1 className="text-3xl font-bold text-gray-800">Technique Analyser AI</h1>
      </header>
      <VideoUpload />
    </div>
  );
}

export default App;
