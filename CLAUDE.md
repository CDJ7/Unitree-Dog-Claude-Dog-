# Unitree-Dog-Claude-Dog: notes for Claude Code

The repository holds three stages of work on a Unitree Go2 (ENG 6010):
- `go2_stream/`: **Test 1**. Streaming, teleop, LiDAR and camera on the real dog. Don't edit without asking.
- `Test2/`: **Test 2**, submitted and **frozen**. Read or import only.
- `Phase_2_Investigator/`: **current work**. The crime-scene investigator, in simulation and on the real dog.

**For anything in Phase 2, read `Phase_2_Investigator/CLAUDE.md` and `Phase_2_Investigator/docs/STATUS.md`
first.** They hold the rules, the status, the next steps and the lessons learned.

**Git:** develop on `claude/eng6010-test2-crime-scene-0rst3r`. The user asks for a PR into `main` after each
milestone and merges it themselves; never merge.

**Results:** never invent numbers; every result quoted must come from a file a script wrote.

**Safety:** never run anything that makes the real dog move. Write the command for the user instead.
