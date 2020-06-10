import React from 'react';
import logo from './logo.svg';
import Playback from './components/Playback';
import './App.css';

window.onbeforeunload = function(){
  return 'Are you sure you want to leave?';
};

function App() {
  return (
    <div className="App">
      <Playback/>
    </div>
  );
}

export default App;
