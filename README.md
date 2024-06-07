# InSights AR

Tartanhacks 2020 Project by Mayank Mali, George Ralph, Audrey Tzeng, Chris Seiler.

![DevPost Blog](https://devpost.com/software/insights-ar)

Winner of Scott Krulcik Grand Prize.

## What it does

InSights AR uses OpenCV for facial recognition and applies it to a real-time video streamed from the Magic Leap headset to recognize people and attach supplemental information.

## How we built it

We used the Magic Leap AR headset and connected it to an Azure server. We used OpenCV and trained models with pictures of faces (obtained with the subject's permission), then sent the classified figures back to the headset to be overlayed on reality. Additionally, we added eye-tracking so the user only sees information for the people they are looking at. This prevents the user from being bombarded with information in a noisy environment. Finally, we integrated a filtering algorithm with a geometrically weighted majority vote system over a buffer history to prevent the classification from wavering frame-to-frame.


