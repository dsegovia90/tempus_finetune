# Labelling spec: time-of-day extraction

The task: read one chat message and list every clock time in it.

Each time has two parts, and times are listed in order of appearance:

- `time` is the time on a 24-hour clock, "HH:MM", from "00:00" to "23:59".
- `text` is copied exactly from the message and is the shortest clock expression: "3", "7:30",
  "4pm", "3 p.m.", "15:00", "noon", "midnight". An am/pm marker is part of it. Context words
  ("in the afternoon"), "ish", and timezone names ("EST") are not.
- A range is two entries: "from noon to 4pm" gives noon and 4pm.

Rules for the time:

1. Read the message as a sensible person would, using world knowledge. "dinner at 7" is 19:00,
   "standup at 9" is 09:00, "got paged at 2, been up since" is 02:00, "let's meet at 3" is
   15:00.
2. Words like morning, afternoon, evening, tonight and night are never times themselves, but
   they decide am or pm for a nearby clock time: "3 in the afternoon" is "3", 15:00.
3. One am/pm marker can cover both ends of a range: "2 to 4pm" is 14:00 and 16:00. Use sense
   when it cannot: "11-1pm" is 11:00 and 13:00.
4. With no context at all (the message is little more than the time itself), assume am: "at 3?"
   is 03:00. A bare 12 is noon: "12" is 12:00, "12:45" is 12:45.
5. "midnight", "12am" and "24:00" are 00:00. "noon" and "12pm" are 12:00.

Not times, never extracted: durations ("a 30 minute call", "ran 2 hours over"), relative times
("in 2 hours", "back in 5"), vague periods alone ("this afternoon"), and numbers that are
something else (counts, rooms, prices, versions, scores, ratios like 16:9, dates, "1:1"
meetings, race results like "finished in 3:45").
