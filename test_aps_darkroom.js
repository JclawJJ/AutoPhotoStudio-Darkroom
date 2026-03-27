const { execSync } = require('child_process');

try {
  console.log("Checking aps-darkroom build...");
  const out = execSync('cd aps-darkroom && npm run build', { encoding: 'utf8' });
  console.log(out.substring(out.length - 200));
} catch (e) {
  console.error("Build failed:", e.stdout.toString(), e.stderr.toString());
}
