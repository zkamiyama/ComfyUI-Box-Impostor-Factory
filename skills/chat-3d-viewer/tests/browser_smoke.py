from pathlib import Path
import os
from playwright.sync_api import sync_playwright
base=Path(__file__).parents[1]/'examples'
for name in ['material_study.inspector.html','walking_dog_sample.inspector.html']:
    html=base/name
    with sync_playwright() as p:
        options={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
        if os.environ.get('CHROME_BIN'):
            options['executable_path']=os.environ['CHROME_BIN']
        browser=p.chromium.launch(**options)
        page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
        errs=[]
        page.on('pageerror',lambda err: errs.append('PAGE '+str(err)))
        page.on('console',lambda msg: errs.append('CONSOLE '+msg.text) if msg.type=='error' else None)
        page.set_content(html.read_text(encoding='utf8'), wait_until='load')
        page.wait_for_timeout(3500)
        print('\n',name,'state=',page.locator('#render-state').inner_text(),'status=',page.locator('#stage-state').inner_text())
        print('stats=',[(s,page.locator('#stat-'+s).inner_text()) for s in ('vertices','triangles','materials','meshes')])
        print('errors',errs)
        page.screenshot(path=str(base/(name.replace('.html','.png'))),full_page=True)
        page.locator('#tab-materials').click()
        print('materials',page.locator('.material-card').count())
        page.screenshot(path=str(base/(name.replace('.html','.materials.png'))),full_page=True)
        page.locator('#tab-textures').click()
        print('textures',page.locator('.texture-option').count())
        page.locator('#channel').select_option('a')
        page.locator('#uv-overlay').check()
        page.screenshot(path=str(base/(name.replace('.html','.textures.png'))),full_page=True)
        page.locator('#tab-details').click()
        page.locator('[data-view="front"]').click()
        page.locator('#wireframe').click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(base/(name.replace('.html','.wire.png'))),full_page=True)
        assert not errs,errs
        assert page.locator('#stat-materials').inner_text()!='—'
        browser.close()
