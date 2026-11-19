#!/usr/bin/env python3
"""Read-only geometric audit of the saved I head meshes; supplemental evidence only."""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path("/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/I-head-local")
REPO = Path("/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory")
spec = importlib.util.spec_from_file_location("head_audit_helper", REPO / "scripts/refine_head_region.py")
head = importlib.util.module_from_spec(spec); spec.loader.exec_module(head)
def read(name): return json.loads((ROOT / name).read_text())
def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
plan, mapping, inference = read("head-plan.json"), read("head-mapping.json"), read("hi3d-inference.json")
cut = float(plan["crop"]["cutPlane"]["value"])
span = float(mapping["nearestVertexDistanceDiagnostic"]["normalizedBySourceYSpan"])
prior_box = np.array([[-0.06190345578729607,0.3100160773953314,-0.024992582091377752],
                      [0.07604182008052954,0.51,0.18067711655172314]])

def quantile(values):
    a = np.asarray(values)
    return np.quantile(a, [0,.5,.95,1]).tolist() if a.size else None

def audit(name, world_frame):
    path = ROOT / name
    data = path.read_bytes()
    doc, binary = head.depth_mesh._decode(data)
    accessor_vertices = sum(len(xyz) for _,xyz,tri,transform in head.depth_mesh._primitives(doc,binary))
    vertices, triangles = head.mesh_arrays(data)
    corners = vertices[triangles]
    cross = np.cross(corners[:,1]-corners[:,0], corners[:,2]-corners[:,0])
    twice_area = np.linalg.norm(cross,axis=1)
    directed = triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    unordered = np.sort(directed,axis=1)
    edges, inverse, counts = np.unique(unordered,axis=0,return_inverse=True,return_counts=True)
    balance = np.bincount(inverse,weights=np.where(directed[:,0] < directed[:,1],1,-1),minlength=len(edges))
    boundary, nonmanifold = edges[counts==1], edges[counts>2]
    _, multiplicity = np.unique(np.sort(triangles,axis=1),axis=0,return_counts=True)
    adjacency = coo_matrix((np.ones(len(edges),dtype="u1"),(edges[:,0],edges[:,1])),
                           shape=(len(vertices),len(vertices))).tocsr()
    ncomponents, labels = connected_components(adjacency,directed=False)
    component_vertices = np.bincount(labels,minlength=ncomponents)
    component_faces = np.bincount(labels[triangles[:,0]],minlength=ncomponents)
    order = np.argsort(component_faces)[::-1]
    components = [{"vertices":int(component_vertices[i]),"triangles":int(component_faces[i]),
                   "triangleFraction":float(component_faces[i]/len(triangles))} for i in order]
    report = {
        "path":str(path),"sha256":sha(path),"fileBytes":path.stat().st_size,
        "activeAccessorVertices":accessor_vertices,"uniquePositions":len(vertices),
        "triangles":len(triangles),"allPositionsFinite":bool(np.isfinite(vertices).all()),
        "allIndicesInBounds":bool(triangles.min()>=0 and triangles.max()<len(vertices)),
        "unreferencedUniquePositions":int(len(vertices)-len(np.unique(triangles))),
        "bounds":[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()],
        "zeroAreaTriangles":int(np.sum(twice_area==0)),
        "repeatedCornerTriangles":int(np.sum(np.any(np.diff(np.sort(triangles,axis=1),axis=1)==0,axis=1))),
        "triangleAreaMinMedianP95Max":quantile(twice_area/2),
        "duplicateUnorientedTriangleGroups":int(np.sum(multiplicity>1)),
        "extraDuplicateUnorientedTriangles":int(np.maximum(multiplicity-1,0).sum()),
        "uniqueEdges":len(edges),"boundaryEdges":len(boundary),"nonmanifoldEdges":len(nonmanifold),
        "maxFacesPerEdge":int(counts.max()),
        "twoFaceEdgesWithSameDirectedWinding":int(np.sum((counts==2)&(balance!=0))),
        "closedTwoManifoldByEdgeIncidence":bool(np.all(counts==2)),
        "vertexConnectedComponents":int(ncomponents),"componentsByTriangleCount":components,
        "eulerVminusEplusF":int(len(vertices)-len(edges)+len(triangles)),
        "eulerDoesNotEstablishGenusWithNonmanifoldEdges":True,
        "selfIntersectionsEvaluated":False,
    }
    if world_frame:
        tolerance=float(max(1.,abs(cut))*np.finfo("f4").eps*2)
        boundary_y=vertices[boundary,1] if len(boundary) else np.empty((0,2))
        midpoints=vertices[nonmanifold].mean(axis=1) if len(nonmanifold) else np.empty((0,3))
        normals=np.divide(cross,twice_area[:,None],out=np.zeros_like(cross),where=twice_area[:,None]>0)
        centroid_y=corners[:,:,1].mean(axis=1)
        lower_slabs={}
        for fraction in (.01,.02,.05):
            selected=centroid_y <= cut+span*fraction
            downward=selected & (normals[:,1]<-.8)
            lower_slabs[str(fraction)]={"upperY":cut+span*fraction,"trianglesByCentroid":int(selected.sum()),
                "downwardFacingTriangles":int(downward.sum()),
                "downwardFacingArea":float(twice_area[downward].sum()/2)}
        report["cutPlane"]={"canonicalY":cut,"planeTolerance":tolerance,
            "verticesBelowPlaneBeyondTolerance":int(np.sum(vertices[:,1]<cut-tolerance)),
            "verticesOnPlaneWithinTolerance":int(np.sum(abs(vertices[:,1]-cut)<=tolerance)),
            "allThreeCornersOnPlaneTriangles":int(np.sum(np.all(abs(corners[:,:,1]-cut)<=tolerance,axis=1))),
            "boundaryEdgesOnPlane":int(np.sum(np.all(abs(boundary_y-cut)<=tolerance,axis=1))),
            "boundaryEndpointYMinMedianP95Max":quantile(boundary_y),
            "nonmanifoldMidpointYMinMedianP95Max":quantile(midpoints[:,1]),
            "nonmanifoldEdgesInLower5PercentSlab":int(np.sum(midpoints[:,1]<=cut+.05*span)),
            "nonmanifoldEdgesInsidePriorHeadQABox":int(np.sum(np.all((midpoints>=prior_box[0])&(midpoints<=prior_box[1]),axis=1))),
            "lowerSlabsByFractionOfSourceYSpan":lower_slabs}
    return report,vertices,triangles

results={}; arrays={}
for role,name,world in [("source","head-source.glb",True),("raw","hi3d-raw.glb",False),
                        ("mapped","head-canonical.glb",True)]:
    results[role],points,triangles=audit(name,world)
    arrays[role]=(points,triangles)
    print(json.dumps({"finished":role,"triangles":len(triangles),"boundary":results[role]["boundaryEdges"],
                      "nonmanifold":results[role]["nonmanifoldEdges"]}),flush=True)
source, source_faces=arrays["source"]
raw, raw_faces=arrays["raw"]
mapped, mapped_faces=arrays["mapped"]
affine=np.array(mapping["mapping"]["rawToCanonicalWorldRowMajor"])
expected=(raw@affine[:3,:3].T+affine[:3,3]).astype("<f4")
mapped_doc,mapped_binary=head.depth_mesh._decode((ROOT/"head-canonical.glb").read_bytes())
mapped_primitives=list(head.depth_mesh._primitives(mapped_doc,mapped_binary))
if len(mapped_primitives)!=1:
    raise AssertionError("Standalone output must have one primitive.")
_,stored_mapped_positions,stored_mapped_faces,stored_transform=mapped_primitives[0]
stored_mapped_world=stored_mapped_positions@stored_transform[:3,:3].T+stored_transform[:3,3]
exact_positions=np.array_equal(expected,stored_mapped_world.astype("<f4"))
distance,_=cKDTree(source).query(mapped,k=1,workers=1)
worst=int(np.argmax(distance)); lower=mapped[:,1]<=cut+.05*span
nearest=mapping["nearestVertexDistanceDiagnostic"]
supplement={
    "valuesInOriginalMapAre":"raw canonical-world coordinate units",
    "normalizedBySourceYSpanMeaning":"a scalar divisor only; stored triplets have NOT already been divided",
    "divisorCanonicalUnits":span,
    "mappedToSourceRawCanonicalUnitsP50P95Max":nearest["mappedToSourceP50P95Max"],
    "sourceToMappedRawCanonicalUnitsP50P95Max":nearest["sourceToMappedP50P95Max"],
    "mappedToSourceFractionsOfYSpanP50P95Max":(np.array(nearest["mappedToSourceP50P95Max"])/span).tolist(),
    "sourceToMappedFractionsOfYSpanP50P95Max":(np.array(nearest["sourceToMappedP50P95Max"])/span).tolist(),
    "metricLimitation":"nearest input vertex, not nearest surface; not a facial-landmark or identity accuracy measurement",
    "worstMappedToSource":{"distanceCanonicalUnits":float(distance[worst]),"mappedVertexXYZ":mapped[worst].tolist(),
                            "inLower5PercentSlab":bool(lower[worst])},
    "mappedToSourceLower5PercentSlabRawMinMedianP95Max":quantile(distance[lower]),
    "mappedToSourceAboveLower5PercentSlabRawMinMedianP95Max":quantile(distance[~lower]),
}
checks={
    "rawHashMatchesMapping":sha(ROOT/"hi3d-raw.glb")==mapping["rawMeshSha256"],
    "mappedHashMatchesMapping":sha(ROOT/"head-canonical.glb")==mapping["outputMeshSha256"],
    "planHashMatchesMapping":sha(ROOT/"head-plan.json")==mapping["planSha256"],
    "inferenceFacesMatchRaw":inference["output_faces"]==len(raw_faces),
    "inferenceVerticesMatchRawAccessors":inference["output_vertices"]==results["raw"]["activeAccessorVertices"],
    "mappedStoredIndicesEqualRawUniquePositionIndices":bool(np.array_equal(raw_faces,stored_mapped_faces)),
    "mappedStoredPositionsEqualDeclaredAffineAtFloat32":bool(exact_positions),
    "coarseWholebodyInputUnchanged":sha(plan["sourceCoarse"]["path"])==plan["sourceCoarse"]["sha256"],
    "fullbodyReferencePNGUnchanged":sha(plan["sourceReference"]["path"])==plan["sourceReference"]["sha256"],
    "sourceCutMeshUnchanged":sha(ROOT/"head-source.glb")==plan["artifacts"]["head-source.glb"]["sha256"],
}
if not all(checks.values()):
    raise AssertionError(json.dumps(checks))
proof={
    "kind":"supplemental-I-head-result-topology-and-mapping-audit-v1",
    "sourcePlan":str(ROOT/"head-plan.json"),"sourcePlanSHA256":sha(ROOT/"head-plan.json"),
    "sourceMapping":str(ROOT/"head-mapping.json"),"sourceMappingSHA256":sha(ROOT/"head-mapping.json"),
    "auditScript":str(Path(__file__).resolve()),"auditScriptSHA256":sha(__file__),
    "helperScript":str(REPO/"scripts/refine_head_region.py"),"helperScriptSHA256":sha(REPO/"scripts/refine_head_region.py"),
    "inputOutputChecks":checks,"meshes":results,"distanceMetricClarification":supplement,
    "priorHeadQABoxCanonical":prior_box.tolist(),
    "cutPlaneInterpretation":"source has an intentionally open plane; generated result reconstructs the lower region. Mapped minimum Y matches the source by bbox-height fit and does not prove anatomical neck correspondence or cut-boundary preservation.",
    "assessment":"finite indexed geometry and declared affine verified; output is not a closed two-manifold by edge incidence. No topology repair, full-body merge, or self-intersection test performed.",
    "diagnosticIndexingNote":"Topology helper sorts unique positions. Float32 rounding can reorder tied XYZ values; exact transform verification therefore uses stored mapped accessor order and indices against the raw unique-position indexing used by map_result.",
    "originalMapEdited":False,"fullBodyMergeAllowed":False
}
target=ROOT/"head-topology.json"
if target.exists():
    raise FileExistsError("Preserve original audit; output already exists: "+str(target))
target.write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
print(json.dumps({"proof":str(target),"bytes":target.stat().st_size,"checks":checks,
                  "components":{k:v["vertexConnectedComponents"] for k,v in results.items()},
                  "worstDistance":supplement["worstMappedToSource"]},indent=2),flush=True)
