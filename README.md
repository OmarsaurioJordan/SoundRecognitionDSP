# SoundRecognitionDSP

A sound recognition with DMNN (neural network), made in Python + Qt

This is a project for a DSP college class, it is a Qt GUI that receive audios to train a neural network and next can detect if a new audio is similar to one of the train ones. For example, the software listen a youtubers voice, train and next detect if other audio is abou that youtuber.

This is educative, so more than the power of the resulting tool, utility or using of great libraries... it only works like an example of DSP system, showing how audio detection works.

In the data include audios to test, some are great in results others poor. All this was made in 2020 with Python 3... by Omwekiatl.

Structure (see the adjunted images):
- A: audio administration, here you can open, save, play, cut, record audio. And extract internal features with 2 possible methods.
- P: patterns administration, you can open, save, delete features of audio classes. And to see classes data, train accuracy and result.
- T: to open, save, create, train and test the network (DMNN)

Usage:
- in A record or open an audio
- in A put a class name you want
- in A clic in one of the 2 methods to extract features
- in P the class name will appear in the list
- repeat to add other audios (minimum should be 2)
- in T clic in the button to create a new net
- in T clic in train the net
- wait a time and finally you see the accuracy info
- in A record or open an audio different to the train ones
- in T clic in test button, same method of extraction
- in P you see the winner class, result in ()

you can download the executable here: https://omwekiatl.itch.io/soundrecognitiondsp
