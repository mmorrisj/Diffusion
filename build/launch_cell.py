# -*- coding: utf-8 -*-
"""The ComfyUI launch cell shared by every ComfyUI notebook in this repo.

Ported from wan22_i2v_comfyui_colab.ipynb (hardened tunnel: same-origin Colab proxy by default,
ngrok or cloudflare as alternatives — avoids ComfyUI v1.19+ 403 host/origin errors)."""

LAUNCH_CELL = r'''
#@title Launch ComfyUI + tunnel { display-mode: "form" }
TUNNEL = "colab"  #@param ["colab", "cloudflare", "ngrok"]
NGROK_TOKEN = ""  #@param {type:"string"}
EXTRA_ARGS = []   # e.g. ['--lowvram'] on a card that OOMs, or ['--reserve-vram', '2']

import os, re, time, subprocess, urllib.request

PORT = 8188
LOG = '/tmp/comfyui.log'

subprocess.run(['pkill', '-f', 'main.py'], capture_output=True)
subprocess.run(['pkill', '-f', 'cloudflared'], capture_output=True)
time.sleep(2)

# --enable-cors-header '*' relaxes cross-origin checks for proxied access.
comfy = subprocess.Popen(
    ['python', 'main.py', '--listen', '127.0.0.1', '--port', str(PORT),
     '--preview-method', 'auto', '--enable-cors-header', '*'] + EXTRA_ARGS,
    cwd='/content/ComfyUI', stdout=open(LOG, 'w'), stderr=subprocess.STDOUT)

print('Waiting for ComfyUI to start...')
ready = False
for _ in range(90):
    time.sleep(2)
    if comfy.poll() is not None:
        print('\n❌ ComfyUI exited. Last log lines:\n')
        print(subprocess.run(['tail', '-n', '40', LOG], capture_output=True, text=True).stdout)
        raise SystemExit('ComfyUI failed to start — see log above.')
    try:
        if urllib.request.urlopen(f'http://127.0.0.1:{PORT}/system_stats', timeout=2).status == 200:
            ready = True
            break
    except Exception:
        pass
if not ready:
    print(subprocess.run(['tail', '-n', '40', LOG], capture_output=True, text=True).stdout)
    raise SystemExit('ComfyUI did not become ready in time — see log above.')
print('✅ ComfyUI is serving on :%d' % PORT)

def banner(url, note=''):
    print('\n' + '=' * 64)
    print('  🚀  Open ComfyUI:  ' + url)
    if note:
        print('  ' + note)
    print('=' * 64)

tunnel = None

if TUNNEL == 'colab':
    # Same-origin embed/link — avoids both the 404 (raw proxy URL pasted in a new
    # browser) and ComfyUI's 403 host/origin check. RECOMMENDED on Colab.
    from google.colab import output
    print('▶ Click this link to open ComfyUI in a new tab:')
    output.serve_kernel_port_as_window(PORT)
    print('\n▶ ...or use ComfyUI embedded right here:')
    output.serve_kernel_port_as_iframe(PORT, height='820')

elif TUNNEL == 'ngrok':
    # ngrok forwards Host == its own domain == Origin, so ComfyUI's host/origin
    # check passes. Reliable public URL that works in any browser.
    if not NGROK_TOKEN:
        raise SystemExit('Set NGROK_TOKEN in the form (free at dashboard.ngrok.com), or use TUNNEL="colab".')
    !pip install -q pyngrok
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = NGROK_TOKEN
    ngrok.kill()
    banner(ngrok.connect(PORT, 'http').public_url, '(ngrok)')

elif TUNNEL == 'cloudflare':
    # Quick tunnel. ComfyUI v1.19+ may 403 ("non matching host and origin")
    # through a proxy; --http-host-header keeps the forwarded Host aligned, and a
    # stale cookie is the other common cause — open the link in an incognito tab.
    !rm -rf ~/.cloudflared
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    !chmod +x /usr/local/bin/cloudflared
    tunnel = subprocess.Popen(
        ['cloudflared', 'tunnel', '--no-autoupdate',
         '--url', f'http://127.0.0.1:{PORT}', '--http-host-header', f'127.0.0.1:{PORT}'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url_re = re.compile(r'https://[a-z0-9\-]+\.trycloudflare\.com')
    found = False
    t0 = time.time()
    for line in tunnel.stdout:
        m = url_re.search(line)
        if m:
            banner(m.group(0), '403? open in an incognito tab; still 403 → use TUNNEL="ngrok" or "colab"')
            found = True
            break
        if time.time() - t0 > 40:
            break
    if not found:
        print('⚠️  Cloudflare returned no URL. Set TUNNEL="colab" or "ngrok" and re-run.')

print('\n⏳ Keep this cell running. Interrupt to stop.')
try:
    comfy.wait()
except KeyboardInterrupt:
    comfy.terminate()
    if tunnel:
        tunnel.terminate()
    print('\n🛑 ComfyUI stopped')
'''
