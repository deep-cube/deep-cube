import './Editor.css';
import React from 'react';
import { DataRaw } from './Playback';

export default function Editor(props: {data: DataRaw, frame: number, offset: number,
    saveTemp: (data: DataRaw) => void, setFrame: (frame: number) => void}) {
    let { offset, frame, data, setFrame } = props
    let [ activeIdx, setActiveIdx ] = React.useState(0)
    let [ editingIdx, setEditingIdx ] = React.useState(-1)

    const activeRef = React.useRef<HTMLDivElement | null>(null)
    // we always show 50 active ones
    React.useEffect( () => {
        if (activeRef.current) {
            activeRef.current.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest',
                inline: 'start'
            })
        }
    }, [activeIdx])

    React.useEffect( () => {
        setEditingIdx(-1)
    }, [data.moves])

    let moves = Object.entries(data.moves)
    let active_i = 0;
    for (let i = 0; i < moves.length; i++) {
        let [moveFrameStr, ] = moves[i]
        if (parseInt(moveFrameStr) + offset >= frame) {
            active_i = i; break;
        }
    }
    if (active_i !== activeIdx)
        setActiveIdx(active_i)

    const handleClick = React.useMemo( () => (idx: number) => {
        setFrame(parseInt(moves[idx][0]) + offset)
        setEditingIdx(idx)
    }, [offset, moves, setFrame])

    let entries = moves.map( ([moveFrameStr, move]: [string, string], i: number) => {
        //let moveDone = frame > (moveFrame + offset);
        //let moveFuture = frame <= (moveFrame + offset - 100);
        //let className = moveDone ? 'inactive' : moveFuture ? 'future' : 'active';
        let timeTilFire = parseInt(moveFrameStr) + offset - frame
        let className = (i !== active_i) ? 'inactive': (timeTilFire < 3) ? 'active' : 'pending';
        return (
            <span
                className={className}
                key={i}
                onClick={() => handleClick(i)}
                ref={ (i === active_i) ? activeRef : null }>
                {move}
            </span>
        )
    })

    let editInfo = (editingIdx === -1) ? "" :
        (offset + parseInt(moves[editingIdx][0])) + " : " + moves[editingIdx][1]

    const handleChange = () => {
        if (editingIdx === -1) return
        let newMove = window.prompt("Change the move to " + editInfo ,moves[editingIdx][1]);
        if (newMove === null) return
        let newData : DataRaw = JSON.parse(JSON.stringify(data)) as DataRaw;
        newData.moves[moves[editingIdx][0]] = newMove;
        props.saveTemp(newData)
    }

    const handleRemove = () => {
        if (editingIdx === -1) return
        let newMove = window.confirm("You want to remove (temporarily set to _, need to save to file to actually delete) ?" + editInfo);
        if (newMove === false) return
        let newData : DataRaw = JSON.parse(JSON.stringify(data)) as DataRaw;
        newData.moves[moves[editingIdx][0]] = "_";
        props.saveTemp(newData)
    }

    const handleAdd = () => {
        let newMove = window.prompt("Add a move to current frame?" +
               ` video_frame=${frame} offset=${offset} target_frame=${frame - offset}` );
        if (newMove === null) return
        let newData : DataRaw = JSON.parse(JSON.stringify(data)) as DataRaw;

        newData.moves[frame - offset] = newMove
        props.saveTemp(newData)
    }

    return <div>
        <div className="entry_container">
            { entries }
        </div>
        <hr></hr>
        <div className="editing_container">
            <div> Editing: { editInfo } </div>
            <button onClick={handleChange}> change </button>
            <button onClick={handleRemove}> remove </button>
            <hr></hr>
            <div> Add a move based on current frame: </div>
            <button onClick={handleAdd}> Add </button>
        </div>
    </div>
}