"""Compare the shipped visual graphs with the node registry of the running ComfyUI."""
import json
from pathlib import Path


def compatible(stored, runtime):
    if isinstance(runtime, list):
        return stored == 'COMBO'
    return bool(set(stored.split(',')) & set(runtime.split(','))) or runtime in ('*', 'COMFY_MATCHTYPE_V3')


def find_input(ports, name):
    if name in ports:
        return ports[name]
    parent, sep, child = name.partition('.')
    if not sep or parent not in ports or ports[parent][0] != 'COMFY_AUTOGROW_V3':
        return None
    template = ports[parent][1]['template']
    names = template.get('names')
    if names is None:
        names = [template['prefix'] + str(i) for i in range(template['max'])]
    if child not in names:
        return None
    for branch in ('required', 'optional'):
        for value in template['input'].get(branch, {}).values():
            return value
    return None


def check_workflows(registry, directory):
    counts = {}
    problems = []
    for path in sorted(Path(directory).glob('*.json')):
        graph = json.loads(path.read_text(encoding='utf-8'))
        definitions = {sg['id']: sg for sg in graph.get('definitions', {}).get('subgraphs', [])}
        count = 0
        for scope in (graph, *definitions.values()):
            for node in scope['nodes']:
                kind = node['type']
                where = f'{path.name}/{scope.get("name", "root")}/{kind}#{node["id"]}'
                if kind in definitions:
                    sg = definitions[kind]
                    for side in ('inputs', 'outputs'):
                        actual = [(p['name'], p['type']) for p in node[side]]
                        expected = [(p['name'], p['type']) for p in sg[side]]
                        if actual != expected:
                            problems.append(f'{where} invalid subgraph {side}')
                    continue
                if kind not in registry:
                    problems.append(f'{where} is missing from /object_info')
                    continue
                info = registry[kind]
                ports = {**info['input'].get('required', {}), **info['input'].get('optional', {})}
                for pin in node.get('inputs', []):
                    value = find_input(ports, pin['name'])
                    if value is None or not compatible(pin['type'], value[0]):
                        problems.append(f'{where} invalid input {pin["name"]} ({pin["type"]})')
                for i, output in enumerate(node.get('outputs', [])):
                    if i >= len(info['output']) or not compatible(output['type'], info['output'][i]):
                        problems.append(f'{where} invalid output {i} ({output["type"]})')
                # Required object sockets must be present; scalar and combo widgets can be literals.
                for k, spec in info['input'].get('required', {}).items():
                    t = spec[0]
                    if isinstance(t, list) or t in ('INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO') or t.startswith('COMFY_'):
                        continue
                    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                    # For example IO.Color.Input is a socketless widget. The official
                    # Pixal graph stores background='#000000' in widgets_values_named.
                    if options.get('socketless') or t in ('COLOR', 'COLORS', 'LOAD_3D'):
                        named = node.get('widgets_values_named') or {}
                        if k not in named and 'default' not in options:
                            problems.append(f'{where} lacks widget value/default for {k}')
                        continue
                    if not any(p['name'] == k and p.get('link') is not None for p in node.get('inputs', [])):
                        problems.append(f'{where} lacks required socket {k}')
                count += 1
        counts[path.name] = count
    if not counts:
        raise ValueError(f'No workflows in {directory}')
    if problems:
        raise AssertionError('\n'.join(problems))
    return counts
