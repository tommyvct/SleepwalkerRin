# Rhythm Game AutoPlayer
## Abstract
Games like Muse Dash, Hatsune Miku Project Diva series, Deemo, Cytus are music games that players are to make interactions according to the rhythm of the music. The goal of the game is to make more timely actions to achieve a higher score. This type of game is highly repetitive so that it makes sense to use machine learning to let the computer play the game by itself. The system which plays such a game automatically is called the AutoPlayer from now on.

## The flow of a rhythm game
The interface of a rhythm game usually have the following part:
- Targets (marked in the green rectangle)
- Notes (marked in the red rectangle)
- Performance Feedback
- Counters  
  This includes score counter, combo counter, health counter, etc.

Let's use Muse Dash as an example:
![](MuseDash.jpg)
The notes will fly in from the right part of the viewport, and move to the targets with a constant velocity. When the center of the note entered the vicinity of the target, the player is expected to hit the corresponding key to register the hit. The time difference between the actual hit and the perfect hit is used to calculate the performance feedback, as the "PERFECT LATE" shown in the screenshot. Music will also be used as an important reference for this timing. These notes are designed right on the beats against the time signature of the music. For example, pop music usually has a 4/4 time signature, then the notes in the game will usually map to every second, fourth, eighth or sixteenth beat. In some rare cases, there may be triplets. 


## How will AutoPlayer do it
One of the many important aspects of how to make an AI is to mimic how a human would do it. 

Practically, my mindset for playing such game will be read the notes from the middle of the screen and encode them in sequence, then leave the timing job to another thread in my brain which map these sequence to the music beats. 

![](architecture.png)

The Computer Vision module will frequently read the viewport of the game. On each frame, matching all the notes and calculate their due timing by their distance from the target, with the help of the extra timing hint from the Audio Analysis module.  Then pack these pieces of information into an object and send them in a queue, while another thread would read the queue and fire keystrokes to the game on time. 
