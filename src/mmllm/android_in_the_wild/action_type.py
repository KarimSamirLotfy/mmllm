# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AndroidInTheWild action types."""

import enum


class ActionType(enum.IntEnum):
  """
  ActionType is an enumeration of integer values representing different types of actions
  supported in the AndroidInTheWild environment.
  Enum Values:
    UNUSED_0 (0): Placeholder for an unused action type.
    UNUSED_1 (1): Placeholder for an unused action type.
    UNUSED_2 (2): Placeholder for an unused action type.
    TYPE (3): Sends text to the emulator without performing clicks for focus or submitting text.
    DUAL_POINT (4): Represents all gesture actions using dual points (e.g., pinch, zoom, click). Clicks are interpreted when the start and end points are the same, while swipes are interpreted when the start and end points differ.
    PRESS_BACK (5): Represents an explicit press of the back button via ADB.
    PRESS_HOME (6): Represents an explicit press of the home button via ADB.
    PRESS_ENTER (7): Represents an ADB command for hitting the enter key.
    UNUSED_8 (8): Placeholder for an unused action type.
    UNUSED_9 (9): Placeholder for an unused action type.
    STATUS_TASK_COMPLETE (10): Indicates the desired task has been completed or is already complete; resets the environment.
    STATUS_TASK_IMPOSSIBLE (11): Indicates the desired task is impossible to complete; resets the environment.
  These action types are used to control and monitor agent interactions and episode status within the AndroidInTheWild environment.
  """



  # Placeholders for unused enum values
  UNUSED_0 = 0
  UNUSED_1 = 1
  UNUSED_2 = 2
  UNUSED_8 = 8
  UNUSED_9 = 9

  ########### Agent actions ###########

  # A type action that sends text to the emulator. Note that this simply sends
  # text and does not perform any clicks for element focus or enter presses for
  # submitting text.
  TYPE = 3

  # The dual point action used to represent all gestures.
  DUAL_POINT = 4

  # These actions differentiate pressing the home and back button from touches.
  # They represent explicit presses of back and home performed using ADB.
  PRESS_BACK = 5
  PRESS_HOME = 6

  # An action representing that ADB command for hitting enter was performed.
  PRESS_ENTER = 7

  ########### Episode status actions ###########

  # An action used to indicate the desired task has been completed and resets
  # the environment. This action should also be used in the case that the task
  # has already been completed and there is nothing to do.
  # e.g. The task is to turn on the Wi-Fi when it is already on
  STATUS_TASK_COMPLETE = 10

  # An action used to indicate that desired task is impossible to complete and
  # resets the environment. This can be a result of many different things
  # including UI changes, Android version differences, etc.
  STATUS_TASK_IMPOSSIBLE = 11
