# Remove the fake DNG export plan for Phase C and D in the markdown based on Claude 3.5's input
sed -i '' 's/Saves: `C_MasterFilter.jpg` & `C_MasterFilter.dng`./Saves: `C_MasterFilter.jpg` with JSON Prompt Metadata./g' /Users/jclaw/.openclaw/workspace/APS_Project/APS_Execution_Plan_Final.md
sed -i '' 's/Saves: `D_Expansion.jpg` & `D_Expansion.dng`./Saves: `D_Expansion.jpg` with Action Ledger JSON./g' /Users/jclaw/.openclaw/workspace/APS_Project/APS_Execution_Plan_Final.md
