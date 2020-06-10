## How to Run the Recording Tool

`npm install`

Then
`npm start`
runs the app in the development mode.<br />

Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

## How to Use the Recording Tool

See [here](https://www.youtube.com/watch?v=bwtSdQeke5w) for a video tutorial on setup.

Hold the cube white facing up and green facing front of you, and click 'calibrate base.'
Then make a x' rotation (now white center should be facing front, and green center facing bottom), and click "calibrate after x' ".
Then undo the x' rotation and do a y rotation. (white facing up, green facing right), and click "calibrate after y' ".

Note that step 2 and step 3 are one-time calibrations. There's no need to repeat these steps if you need to reset the base orientation again -- simply do step 1 'calibrate base' would do.

## Description

This tool outputs the raw data faithfully from the emitted events, without merging events. Each event contains a timestamp and a string describing the move. There are two types of moves: a turn of a layer that is one of "UDFBLR", or a whole-cube rotation around a single axis that is one of "xyz". Our notation follows the standard Rubik's Cube notation. In our case, due to how the sensor works, layer turns can only be quarter turns. 

As an exception, if an orientation change cannot be explained by a simple rotation x*, y*, z*, (i.e we rotate too quickly or intentionally perform a combo move), I'll output the compound rotation in one event, using one of the many possible sequence of basic moves that is equivalent to the compound rotation. ( e.g. An "x2 y" event does NOT mean x2 y is the actual sequence of rotations that happened -- the user can rotate anyhow they like). 

Because each timeframe may see multiple events fired, moves are represented as tuples sorted by time. The schema is given below.

## Schema

```
  data := {
      moves: [ delta_time_relative_to_first_move_in_millisecond, move_string ][]
  }
  move_string := layer_turn | rotations
  layer_turn := layer + layer_amount
  layer := U | F | R | D | B | L
  layer_amount := (none) | '
  
  rotations := rotation | rotation + " " + rotation
  rotation := axis + axis_amount
  axis := x | y | z
  axis_amount := (none) | 2 | ' 
```
