# Hooking up the real Python scripts

We have the `aps_server.py` FastAPI mock. Let's make it real.
1. `aps_server.py` must import Phase 1 Mapper and Phase 2 Sorter and Phase 3 Forge.
2. In `POST /api/process`, take the image file, run the real Sorter. If bad score, return 400 'Rejected'.
3. Run the real Mapper. Generate the mask png locally.
4. Return the real path to the mask and the image as a payload to the frontend.
5. (Do NOT call the ComfyUI forge yet since it requires an external process, just log it and fake the "after" image by passing the original). 

Modify `aps_server.py` properly. Test the python build to make sure `python3 aps_server.py` works and `requests/opencv/mediapipe` doesn't crash it.
