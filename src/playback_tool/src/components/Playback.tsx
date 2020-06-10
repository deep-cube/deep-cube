import './Playback.css';

import { Cube } from '../lib/CubeLib';
import CubeSim from './CubeSim';
import React from 'react';
import Editor from './Editor';

type Move = [number, string];

export type DataRaw = {
  moves: { [key: string]: string };
};
const emptyData: DataRaw = {
  moves: {},
};

const FRAMERATE = 30;
const frameTime = 1 / FRAMERATE;

function timelineView(data: DataRaw, frame: number, offset: number) {
  let movestr = '';
  for (let i = frame - 10; i <= frame + 10; i++) {
    let fs = `${i - offset}`;
    if (data.moves[fs]) {
      let move = data.moves[fs];
      movestr += move;
      if (move.length == 1) {
        movestr += '_';
      }
    } else {
      movestr += '__';
    }
  }
  return (
    <div className='dataContainer'>
      <div> . . . . . . . . . . | . . . . . . . . . . </div>
      <div>{movestr}</div>
    </div>
  );
}

type PlaybackState = {
  data: DataRaw;
  frame: number;
  time: number;
  fileList: string[];
  chosenFilePrefix: string;
  offset: number;
  offsetAbsoluteStr: string;
  labelLoadFrom: 'label' | 'label_aligned';
  recentSaved: boolean;
  playbackRate: number;
};

class Playback extends React.Component<{}, PlaybackState> {
  dataFileInput = React.createRef<HTMLInputElement>();
  videoFileInput = React.createRef<HTMLInputElement>();
  videoRef = React.createRef<HTMLVideoElement>();

  state: PlaybackState = {
    data: emptyData,
    frame: 0,
    time: 0.0,
    fileList: [],
    chosenFilePrefix: '',
    offset: 0,
    offsetAbsoluteStr: '',
    labelLoadFrom: 'label',
    recentSaved: true,
    playbackRate: 1.0
  };

  componentDidMount() {
    this.queryFileList();
    this.handleVideoTimeUpdate();
  }

  queryFileList() {
    let xhr = new XMLHttpRequest();
    xhr.addEventListener('load', () => {
      this.setState({ fileList: JSON.parse(xhr.response) });
    });
    xhr.open('GET', 'http://localhost:12345/list_filenames');
    xhr.send();
  }

  queryVideoAndLabel(filePrefix: string) {
    let labelxhr = new XMLHttpRequest();
    labelxhr.addEventListener('load', () => {
      let data = JSON.parse(labelxhr.response);
      this.setState({
        data: data,
        offsetAbsoluteStr: "",
        recentSaved: true
      });
    });
    labelxhr.open(
      'GET',
      `http://localhost:12345/get_file/${this.state.labelLoadFrom}/${filePrefix}.json`
    );
    labelxhr.send();

    let videoxhr = new XMLHttpRequest();
    videoxhr.addEventListener('load', () => {
      let videofile = new File([videoxhr.response], 'videofile');
      let videourl = URL.createObjectURL(videofile);
      console.log(videourl);
      if (this.videoRef.current) {
        this.videoRef.current.src = videourl;
        this.videoRef.current.load();
        this.videoRef.current.playbackRate = this.state.playbackRate
      }
    });
    videoxhr.responseType = 'blob';
    videoxhr.open(
      'GET',
      `http://localhost:12345/get_file/video/${filePrefix}.mp4`
    );
    videoxhr.send();
  }

  handleVideoTimeUpdate = () => {
    const func = () => {
      if (this.videoRef.current === null) return;
      let video = this.videoRef.current;
      if (!video) return;
      const frame = (video.currentTime * FRAMERATE) | 0;
      this.setState({
        time: video.currentTime,
        frame: frame,
      });
    }
    func();
    setTimeout(this.handleVideoTimeUpdate, 30);
  };

  gotoFrame = (offsetFrame: number) => {
    if (this.videoRef.current === null) return;
    let video = this.videoRef.current;
    if (!video) return;
    video.pause();
    video.currentTime = Math.min(
      video.duration,
      video.currentTime + frameTime * offsetFrame
    );
  };

  increaseSpeed = (deltaSpeed: number) => {
    if (this.videoRef.current === null) return;
    let video = this.videoRef.current
    if (!video) return;
    let newRate = this.state.playbackRate + deltaSpeed;
    this.setState({playbackRate: newRate});
    video.playbackRate = newRate;
  }

  gotoFrameAbsolute = (offsetFrame: number) => {
    if (this.videoRef.current === null) return;
    let video = this.videoRef.current;
    if (!video) return;
    video.pause();
    video.currentTime = Math.min(
      video.duration,
      frameTime * offsetFrame
    );
  };

  offsetSequence = (offset: number) => {
    this.setState((os) => ({
      ...os,
      offset: os.offset + offset,
      offsetAbsoluteStr: (os.offset + offset).toString()
    }));
  };

  offsetSequenceAbsolute = (offset: number) => {
    this.setState((os) => ({
      ...os,
      offset: offset,
    }));
  };

  getDatafileList = () => {
    return this.state.fileList.map((filePrefix: string) => (
      <div
        className='clickable'
        key={`filename-${filePrefix}`}
        onClick={() => {
          this.setState({ chosenFilePrefix: filePrefix });
          this.queryVideoAndLabel(filePrefix);
        }}>
        {filePrefix}
      </div>
    ));
  };

  getMoveList = () => {
    // return Object.keys(this.state.data.moves).map((k: string) => {
    //   let framenum = parseInt(k);
    //   let moveDone = this.state.frame - this.state.offset >= framenum;
    //   return (
    //     <div
    //       className='clickable'
    //       key={`move${framenum}`}
    //       // style={{
    //       //   color: moveDone ? 'red' : 'black',
    //       //   fontWeight: moveDone ? 'bold' : 'normal',
    //       // }}
    //       onClick={() => {
    //         this.gotoFrame(framenum + this.state.offset - this.state.frame);
    //       }}>
    //       {framenum + this.state.offset} : {this.state.data.moves[framenum]}
    //     </div>
    //   );
    // });
  };

  getCube = () => {
    let moveSoFar = '';
    for (let [f, m] of Object.entries(this.state.data.moves)) {
      let fn = parseInt(f) + this.state.offset;
      if (fn <= this.state.frame) moveSoFar += m;
    }
    return Cube.to_facelet(Cube.create(moveSoFar));
  };

  saveTemp = (newData: DataRaw) => {
    this.setState({
      data: newData,
      recentSaved: false
    })
  }

  saveLabel = () => {
    let shifted_data: DataRaw = { moves: {} };

    // _ is placeholder for pending deletes. Remove these keys.
    for (let k in this.state.data.moves) {
      if (this.state.data.moves[k] === "_") {
        delete this.state.data.moves[k]
      }
    }
    for (let k in this.state.data.moves) {
      shifted_data.moves[
        parseInt(k) + this.state.offset
      ] = this.state.data.moves[k];
    }


    let videoxhr = new XMLHttpRequest();
    videoxhr.responseType = 'blob';
    videoxhr.open(
      'POST',
      `http://localhost:12345/save_file/label_aligned/${this.state.chosenFilePrefix}.json`
    );
    videoxhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
    videoxhr.send(JSON.stringify(shifted_data));
    videoxhr.onload = () => {
      alert(`Save request is successful: http://localhost:12345/save_file/label_aligned/${this.state.chosenFilePrefix}.json`);
      this.setState({
        recentSaved: true
      })
    }
  };

  render() {
    return (
      <div className='main'>
        <div className='containerLeft'>
          {this.state.chosenFilePrefix && (
            <div>
              <video
                controls
                height={400}
                ref={this.videoRef}
                // onTimeUpdate={this.handleVideoTimeUpdate}
              />

              <div className='dataContainer'>
                <div>current time: {this.state.time}</div>
                <div>current frame: {this.state.frame}</div>
                <div>current speed: {this.state.playbackRate}x</div>

                <button
                  onClick={() => {
                    this.gotoFrame(-5);
                  }}>
                  -5 frame
                </button>

                <button
                  onClick={() => {
                    this.gotoFrame(-1);
                  }}>
                  -1 frame
                </button>

                <button
                  onClick={() => {
                    this.gotoFrame(1);
                  }}>
                  +1 frame
                </button>

                <button
                  onClick={() => {
                    this.gotoFrame(5);
                  }}>
                  +5 frame
                </button>

                <button
                  onClick={() => {
                    this.increaseSpeed(-0.25);
                  }}>
                  -0.25x playback speed
                </button>

                <button
                  onClick={() => {
                    this.increaseSpeed(0.25);
                  }}>
                  +0.25x playback speed
                </button>

              </div>

              <div className='cubeContainer'>
                <CubeSim height={300} width={300} cube={this.getCube()} />
              </div>
            </div>
          )}

          <div className='dataContainer'>
            <h3>datafiles</h3>
            <div>
              loading from directory: {this.state.labelLoadFrom}
              <button
                onClick={() => {
                  this.setState({
                    labelLoadFrom:
                      this.state.labelLoadFrom === 'label'
                        ? 'label_aligned'
                        : 'label',
                  });
                }}>
                toggle
              </button>
            </div>
            <p>click to load</p>
            {this.getDatafileList()}
            <br />
          </div>
        </div>

        <div className='containerRight'>
          <div className='dataContainer'>
            <h3>moves</h3>

            <div className='unsaveHint'>
              {this.state.recentSaved ? "" : "You have changes unsaved! "}
            </div>
            {/* <p>click to go to end of move</p> */}

            <div>Curr offset amount = {this.state.offset}</div>
            <div>
              <button
                onClick={() => {
                  this.offsetSequence(-10);
                }}>
                shift label right {'-10<<'}
              </button>
              <button
                onClick={() => {
                  this.offsetSequence(-1);
                }}>
                {'<<'} shift label left
              </button>

              <button
                onClick={() => {
                  this.offsetSequence(1);
                }}>
                shift label right {'>>'}
              </button>
              <button
                onClick={() => {
                  this.offsetSequence(10);
                }}>
                shift label right {'>>10'}
              </button>
            </div>
            <div>
              <button
                onClick={() => {
                  this.offsetSequenceAbsolute(parseInt(this.state.offsetAbsoluteStr));
                }}>
                Set offset to input
              </button>
              <input value={this.state.offsetAbsoluteStr}
                onChange={(e: { target: { value: any; }; }) =>
                  this.setState({offsetAbsoluteStr: e.target.value}) } />


            </div>
            <button
              onClick={this.saveLabel} >
              {' '}
              save to label_aligned/{this.state.chosenFilePrefix}.json
            </button>
            <br />
            Relative timeline view
            {timelineView(this.state.data, this.state.frame, this.state.offset)}
            <br />
            {this.getMoveList()}
          </div>

          <div className='editorContainer'>
            <Editor data={this.state.data} frame={this.state.frame}
              offset={this.state.offset} saveTemp={this.saveTemp}
              setFrame={this.gotoFrameAbsolute}
              />
          </div>
        </div>
      </div>
    );
  }
}

export default Playback;

// function getStrings(data: Data, currFrame: number) {
//   let str_l = '',
//     str_curr = '',
//     str_r = '';

//   const duration = 15;
//   let frame_l = Math.max(0, currFrame - duration);
//   let frame_r = Math.min(data.num_frames, currFrame);
//   let first_within = true;
//   for (const [k, r] of data.moves) {
//     if (k < frame_l) {
//       str_l += r;
//     } else if (frame_l <= k && k < frame_r && first_within) {
//       first_within = false;
//       str_curr = r;
//     } else {
//       str_r += r;
//     }
//   }
//   return [str_l, str_curr, str_r];
// }

// function CurrentMoveView(props: { strs: string[] }) {
//   let [str_l, str_curr, str_r] = props.strs;
//   return (
//     <div className='moveContainer'>
//       {str_l.split('').map((ch, i) => {
//         return (
//           <span key={i} className='moveText'>
//             {' '}
//             {ch}{' '}
//           </span>
//         );
//       })}
//       {str_curr.split('').map((ch, i) => {
//         return (
//           <span key={i + str_l.length} className='moveTextPrimary'>
//             {' '}
//             {ch}{' '}
//           </span>
//         );
//       })}
//       {str_r.split('').map((ch, i) => {
//         return (
//           <span key={i + str_l.length + str_curr.length} className='moveText'>
//             {' '}
//             {ch}{' '}
//           </span>
//         );
//       })}
//     </div>
//   );
// }
