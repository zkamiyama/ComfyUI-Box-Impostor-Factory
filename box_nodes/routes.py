"""Serve the package viewer and explicitly declared asset members on Comfy's origin."""
import json
from pathlib import Path

from aiohttp import web

from . import store


def visible_members(m):
    members = {'manifest.json', 'previews/hero.png', m['proxy'], m['baseColor']['path']}
    members.update(d['path'] for d in m['textures'].values())
    for face in m['faces'].values():
        members.add(face['guide'])
        members.add(face.get('depth'))
        members.add(face.get('edit', {}).get('view'))
        members.add(face.get('edit', {}).get('reference'))
    if m.get('reference'):
        members.add(m['reference'].get('image'))
        members.add(m['reference'].get('camera', {}).get('image'))
    members.add(m.get('qa'))
    return members


def register_routes():
    from server import PromptServer
    routes = PromptServer.instance.routes

    @routes.get('/box-impostor/assets/{asset_id}/{relative:.+}')
    async def asset(request):
        try:
            directory = store.asset_dir(request.match_info['asset_id'])
            member = request.match_info['relative']
            path = store.contained(directory, member)
            manifest = json.loads(store.contained(directory, 'manifest.json').read_bytes())
            if member not in visible_members(manifest) or not path.is_file():
                raise web.HTTPNotFound()
        except (ValueError, KeyError, TypeError, OSError):
            raise web.HTTPNotFound()
        media_type = {'.json': 'application/json', '.png': 'image/png', '.glb': 'model/gltf-binary'}.get(path.suffix.lower(), 'application/octet-stream')
        return web.FileResponse(path, headers={'Content-Type': media_type, 'X-Content-Type-Options': 'nosniff'})

    @routes.get('/box-impostor/{section:web|worker|vendor}/{relative:.+}')
    async def static(request):
        try:
            section, relative = request.match_info['section'], request.match_info['relative']
            if section == 'web' and relative == 'viewer.html':
                html = (store.REPO / 'web' / 'viewer.html').read_text(encoding='utf-8').replace('__BI_BASE__', '/box-impostor')
                return web.Response(text=html, content_type='text/html')
            if section == 'vendor':
                valid = (relative.startswith(('three/build/', 'three/examples/jsm/')) or relative == 'three-impostor/Impostor.js') and relative.endswith('.js')
                directory = store.REPO / 'node_modules'
            else:
                valid = relative.endswith('.mjs') and '/' not in relative and relative not in ('cli.mjs', 'browser_bridge.mjs')
                directory = store.REPO / section
            path = store.contained(directory, relative)
            if not valid or not path.is_file():
                raise web.HTTPNotFound()
        except (ValueError, KeyError, OSError):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Content-Type': 'application/javascript', 'X-Content-Type-Options': 'nosniff'})
