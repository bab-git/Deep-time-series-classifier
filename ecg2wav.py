#!/usr/bin/python3
#
# Q & D script to convert .ecg files from Charite to .wav format.
# The .wav format is chosen because it can easily be viewed in tools
# like Audacity or imported in tools like Matlab.
#
# (C)opyright 2019 Fraunhofer IMS
#
# Author: Kai Grundmann
#
# Parameters: Path of .wav file
# Reads from stdin

import sys
import re
import wave

# Read ecg data from stdin
#ecgFile = sys.stdin.buffer.read()

def ecg2wave_conv(in_path,out_path, verbose = None):
    ecgfile  = open(in_path, "rb", buffering = -1)
    ecgFile  = ecgfile.read()
    #ecgFile = path
    
    
    # Find index of first null byte. This seperates the text header from
    # binary data.
    startOfBody = 0
    while ecgFile [startOfBody] != 0:
      startOfBody = startOfBody + 1
    startOfBody = startOfBody + 1
    
    # Get the text header
    header = ""
    for i in range (0, startOfBody):
      header += chr (ecgFile [i])
    
    # Parse the text format header
    # Most of this will not be used, but it may be a good idea to have
    # these data at hand for later purposes.
    m = re.search ('Typ: (.*)', header)
    headerTyp = str (m.group (1))
    if verbose:
        print ("Typ: " + headerTyp, file=sys.stderr)
    
    m = re.search ('Sample Rate: (.*)', header)
    headerSampleRate = float (m.group (1))
    if verbose:
        print ("Sample rate: " + str (headerSampleRate), file=sys.stderr)
    
    m = re.search ('Channels: (.*)', header)
    headerChannels = int (m.group (1))
    if verbose:
        print ("Channels: " + str (headerChannels), file=sys.stderr)
    
    m = re.search ('Channel Names: (.*)', header)
    headerChannelNames = str (m.group (1))
    if verbose:
        print ("Channel names: " + headerChannelNames, file=sys.stderr)
    
    m = re.search ('Factor: (.*)', header)
    headerFactor = float (m.group (1))
    if verbose:
        print ("Factor: " + str (headerFactor), file=sys.stderr)
    
    m = re.search ('Offset: (.*)', header)
    headerOffset = float (m.group (1))
    if verbose:
        print ("Offset: " + str (headerOffset), file=sys.stderr)    
        print ("", file=sys.stderr)
    
    # Do some sanity checking
    if (headerChannels != 2):
      print ("ERROR: Sanity check failed, channels != 2", file=sys.stderr)
      exit
    
    if (headerSampleRate != 512):
      print ("ERROR: Sanity check failed, sample rate != 512", file=sys.stderr)
      exit
    
    if (headerChannelNames != "I III\r"):
      print ("ERROR: Sanity check failed, channel names != I III", file=sys.stderr)
      exit
    
    # Determine the length of the body.
    # The data in the body is 16 bit little endian, channels interleaved.
    # One channel may be a sample longer than the other one.
    bodyLen = len (ecgFile) - startOfBody
    
    if ((int (bodyLen / 2) * 2) != bodyLen):
      print ("ERROR: Sanity check failed, body length not divisible by 2", file=sys.stderr)
      exit
    
    # Get the channel data.
    samples = bytearray ()
    
    i = 0
    while (i < bodyLen):
      samples.append (ecgFile [i + startOfBody])
      i = i + 1
      samples.append (ecgFile [i + startOfBody])
      i = i + 1
      if (i < bodyLen):
        samples.append (ecgFile [i + startOfBody])
        i = i + 1
        samples.append (ecgFile [i + startOfBody])
        i = i + 1
      else:
        samples.append (0)
        samples.append (0)
    
    # Output wave file
    #wf = wave.open (sys.argv [1], 'wb')
#    if save_out:
    wf = wave.open (out_path[:-4]+'.wav', 'wb')
    wf.setnchannels (2)
    wf.setsampwidth (2)
    wf.setframerate (headerSampleRate)
    wf.writeframesraw (samples)
    wf.close()
#    else:
#        return samples
