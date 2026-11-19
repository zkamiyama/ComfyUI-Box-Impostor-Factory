
(() => {
  const probe = globalThis.__inspectorProbe = {factoryCalls: 0, renderCalls: 0, errors: []};
  let wrapped;
  Object.defineProperty(globalThis, 'InspectorThree', {
    configurable: true,
    get() { return wrapped; },
    set(value) {
      const originalFactory = value.createThreeViewer;
      wrapped = new Proxy(value, {
        get(target, key, receiver) {
          if (key !== 'createThreeViewer') return Reflect.get(target, key, receiver);
          return async function(...args) {
            probe.factoryCalls++;
            const scene = args[0]?.scene;
            probe.scene = {bounds: scene?.bounds, center: scene?.center,
                           radius: scene?.radius, assetId: scene?.assetId};
            try {
              const api = await Reflect.apply(originalFactory, target, args);
              probe.api = api;
              probe.apiKeys = Object.keys(api);
              const renderer = api.getRenderer();
              const originalRender = renderer.render;
              renderer.render = function(scene, camera, ...tail) {
                const result = Reflect.apply(originalRender, this, [scene, camera, ...tail]);
                probe.renderCalls++;
                probe.cameraObject = camera;
                probe.camera = {
                  type: camera.type,
                  position: camera.position.toArray(), quaternion: camera.quaternion.toArray(),
                  up: camera.up.toArray(), matrixWorld: camera.matrixWorld.toArray(),
                  matrixWorldInverse: camera.matrixWorldInverse.toArray(),
                  projectionMatrix: camera.projectionMatrix.toArray(),
                  projectionMatrixInverse: camera.projectionMatrixInverse.toArray(),
                  near: camera.near, far: camera.far, left: camera.left, right: camera.right,
                  top: camera.top, bottom: camera.bottom, zoom: camera.zoom, fov: camera.fov,
                  canvasPixels: [renderer.domElement.width, renderer.domElement.height]
                };
                return result;
              };
              return api;
            } catch (error) {
              probe.errors.push(String(error));
              throw error;
            }
          };
        }
      });
    }
  });
})();
