# karaokify

a python desktop app that turns spotify into a real-time karaoke machine using local ai source separation.

## how it works

it captures audio output from a virtual audio cable (like VB-Audio Cable), passes it through a 2-stem spleeter model in sliding chunks to strip out the vocals, and sends the resulting instrumental track straight to your headphones or speakers in real time.

## prerequisites

you will need python 3.9 and a virtual audio device configured on your machine to route audio from spotify into the app.

```bash
pip install -r requirements.txt
```

## running the app

you can run the application either through the graphical interface or via the command line:

```bash
python ui.py
```

or 

```bash
python terminal.py
```
