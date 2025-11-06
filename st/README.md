
# EMG-Controlled Haptic Feedback System

## Overview
This README file contains important information related to the setup of the EMG sensor, Arduino configuration, haptic feedback, and user interface for this Unity project.

## Arduino Configuration
- **EMG Input:** Connect the EMG sensor to Arduino pin A0.
- **Haptic Feedback:** Connect the haptic feedback device to Arduino pin D7.

## Arduino Connection
- **Port:** COM4
- **Baud Rate:** 115200

## Threshold Setup
**Note:** Threshold calibration is required for each scene for consistency.
- Press `R` to set the right hand to correspond to the EMG (this will render the right hand as mechanical).
- Press `L` to set the left hand to correspond to the EMG (this will render the left hand as mechanical).
- Press `C` to calibrate the EMG threshold. This calibration involves taking the average of the input value over a 3-second duration (duration can be adjusted).
