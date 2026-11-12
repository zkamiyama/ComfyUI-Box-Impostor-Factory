export const directions = [
  ['+X',1,0,0], ['-X',-1,0,0], ['+Y',0,1,0], ['-Y',0,-1,0], ['+Z',0,0,1], ['-Z',0,0,-1]
];
for (const axis of ['xy','xz','yz']) for (const a of [-1,1]) for (const b of [-1,1]) {
  const xyz = [0,0,0];
  xyz['xyz'.indexOf(axis[0])] = a;
  xyz['xyz'.indexOf(axis[1])] = b;
  directions.push([axis + ':' + a + ',' + b, ...xyz]);
}
