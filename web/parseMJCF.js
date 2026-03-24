/**
 * Parse a MuJoCo MJCF XML string into the humanoidData structure used by the web demo.
 * Produces: bodies, joints, fixedJoints, dofInfo, kinematicJoints,
 *           fk_parent_indices, fk_local_translations, fk_local_rotations, actuatorOrder.
 *
 * Model-specific data (obs_mean/std, init_dof_pos, action bounds, etc.)
 * must be provided separately via a config object.
 *
 * Usage:
 *   const xml = await (await fetch('humanoid.xml')).text();
 *   const mjcfData = parseMJCF(xml, { fixedBodies: new Set(['sword','shield','left_hand']) });
 *   Object.assign(humanoidData, mjcfData);
 */

// ── Vector / quaternion math ────────────────────────────────────────────────

function _normalize(v) {
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return n < 1e-12 ? v : v.map(x => x / n);
}

function _cross(a, b) {
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}

function _dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

function _getRotationQuat(from, to) {
    const u = _normalize(from), v = _normalize(to);
    const d = _dot(u, v);
    if (d > 1 - 1e-6) return [0, 0, 0, 1]; // identity xyzw
    if (d < 1e-6 - 1) {
        let axis = _cross([1,0,0], u);
        if (_dot(axis, axis) < 1e-6) axis = _cross([0,1,0], u);
        axis = _normalize(axis);
        return [axis[0], axis[1], axis[2], 0]; // 180 deg
    }
    const c = _cross(u, v);
    const s = Math.sqrt((1 + d) * 2), inv = 1 / s;
    const q = [c[0]*inv, c[1]*inv, c[2]*inv, 0.5*s];
    const qn = Math.sqrt(q.reduce((s, x) => s + x * x, 0));
    return q.map(x => x / qn);
}

function _quatRotate(q, v) {
    const qv = [q[0],q[1],q[2]], qw = q[3];
    const t = [2*(qv[1]*v[2]-qv[2]*v[1]), 2*(qv[2]*v[0]-qv[0]*v[2]), 2*(qv[0]*v[1]-qv[1]*v[0])];
    return [v[0]+qw*t[0]+qv[1]*t[2]-qv[2]*t[1],
            v[1]+qw*t[1]+qv[2]*t[0]-qv[0]*t[2],
            v[2]+qw*t[2]+qv[0]*t[1]-qv[1]*t[0]];
}

function _mat33ToQuat(cols) {
    const [m00,m10,m20] = cols[0], [m01,m11,m21] = cols[1], [m02,m12,m22] = cols[2];
    const tr = m00 + m11 + m22;
    let x, y, z, w;
    if (tr >= 0) {
        const h = Math.sqrt(tr + 1); w = 0.5*h; const f = 0.5/h;
        x = (m21-m12)*f; y = (m02-m20)*f; z = (m10-m01)*f;
    } else {
        let i = 0;
        if (m11 > m00) i = 1;
        if (m22 > [m00,m11,m22][i]) i = 2;
        if (i === 0) {
            const h = Math.sqrt(m00-m11-m22+1); x = 0.5*h; const f = 0.5/h;
            y = (m01+m10)*f; z = (m20+m02)*f; w = (m21-m12)*f;
        } else if (i === 1) {
            const h = Math.sqrt(m11-m22-m00+1); y = 0.5*h; const f = 0.5/h;
            z = (m12+m21)*f; x = (m01+m10)*f; w = (m02-m20)*f;
        } else {
            const h = Math.sqrt(m22-m00-m11+1); z = 0.5*h; const f = 0.5/h;
            x = (m20+m02)*f; y = (m12+m21)*f; w = (m10-m01)*f;
        }
    }
    const q = [x,y,z,w], n = Math.sqrt(q.reduce((s,v)=>s+v*v,0));
    return q.map(v => v/n);
}

function _computeJointFrame(jointAxes) {
    const axisMap = [0, 1, 2];
    const n = jointAxes.length;
    if (n === 0) return { q: [0,0,0,1], axisMap };
    if (n === 1) return { q: _getRotationQuat([1,0,0], jointAxes[0]), axisMap };

    const Q = _getRotationQuat(jointAxes[0], [1,0,0]);
    const b = _normalize(_quatRotate(Q, jointAxes[1]));

    if (n === 2) {
        if (Math.abs(_dot(b,[0,1,0])) > Math.abs(_dot(b,[0,0,1]))) {
            axisMap[1] = 1;
            const c = _normalize(_cross(jointAxes[0], jointAxes[1]));
            return { q: _mat33ToQuat([_normalize(jointAxes[0]), _normalize(jointAxes[1]), c]), axisMap };
        } else {
            axisMap[1] = 2; axisMap[2] = 1;
            const c = _normalize(_cross(jointAxes[1], jointAxes[0]));
            return { q: _mat33ToQuat([_normalize(jointAxes[0]), c, _normalize(jointAxes[1])]), axisMap };
        }
    }
    // n === 3
    if (Math.abs(_dot(b,[0,1,0])) > Math.abs(_dot(b,[0,0,1]))) {
        axisMap[1] = 1; axisMap[2] = 2;
        return { q: _mat33ToQuat([_normalize(jointAxes[0]), _normalize(jointAxes[1]), _normalize(jointAxes[2])]), axisMap };
    } else {
        axisMap[1] = 2; axisMap[2] = 1;
        return { q: _mat33ToQuat([_normalize(jointAxes[0]), _normalize(jointAxes[2]), _normalize(jointAxes[1])]), axisMap };
    }
}

// ── MJCF Parser ─────────────────────────────────────────────────────────────

function _parseVec(s, n) {
    if (!s) return null;
    const parts = s.trim().split(/\s+/).map(Number);
    return n ? parts.slice(0, n) : parts;
}

/**
 * Parse MJCF XML and return the humanoidData skeleton structure.
 * @param {string} xmlText - raw MJCF XML string
 * @param {Object} opts
 * @param {Set<string>} [opts.fixedBodies] - body names with fixed joints (auto-detected if omitted)
 * @returns {Object} Partial humanoidData (skeleton/joints/FK — no normalizer or model data)
 */
function parseMJCF(xmlText, opts = {}) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlText, 'text/xml');
    const root = doc.documentElement; // <mujoco>

    // Auto-detect fixed bodies if not provided
    let fixedBodies = opts.fixedBodies;
    if (!fixedBodies) {
        fixedBodies = new Set();
        const scan = (el, isRoot) => {
            if (!isRoot) {
                const hinges = [...el.querySelectorAll(':scope > joint')].filter(
                    j => (j.getAttribute('type') || 'hinge') !== 'free');
                const freeJ = el.querySelector(':scope > freejoint');
                if (!hinges.length && !freeJ) fixedBodies.add(el.getAttribute('name'));
            }
            for (const child of el.querySelectorAll(':scope > body')) scan(child, false);
        };
        for (const b of root.querySelector('worldbody').querySelectorAll(':scope > body'))
            scan(b, true);
    }

    // Actuator map + order
    const actuatorMap = {};
    const actuatorOrder = [];
    const actuatorSec = root.querySelector('actuator');
    if (actuatorSec) {
        for (const mot of actuatorSec.querySelectorAll('motor')) {
            const jname = mot.getAttribute('joint');
            if (!jname) continue;
            const gear = parseFloat(mot.getAttribute('gear') || '1');
            const frc = _parseVec(mot.getAttribute('actuatorfrcrange'), 2);
            const maxForce = frc ? Math.max(Math.abs(frc[0]), Math.abs(frc[1])) : gear;
            actuatorMap[jname] = { gear, maxForce };
            actuatorOrder.push(jname);
        }
    }

    const bodies = [], joints = [], fixedJoints = [];

    function processBody(el, parentName, parentWorldPos) {
        const name = el.getAttribute('name');
        const localPos = _parseVec(el.getAttribute('pos'), 3) || [0,0,0];
        const worldPos = localPos.map((v, i) => parentWorldPos[i] + v);

        // Geoms
        const geoms = [];
        for (const ge of el.querySelectorAll(':scope > geom')) {
            const g = { name: ge.getAttribute('name') || name, type: ge.getAttribute('type') || 'sphere' };
            g.pos = _parseVec(ge.getAttribute('pos'), 3) || [0,0,0];
            if (ge.getAttribute('size')) g.size = _parseVec(ge.getAttribute('size'));
            if (g.type === 'sphere') g.radius = g.size[0];
            else if (g.type === 'capsule') {
                g.radius = g.size[0];
                if (ge.getAttribute('fromto')) g.fromto = _parseVec(ge.getAttribute('fromto'), 6);
            } else if (g.type === 'box') {
                g.halfExtents = g.size.slice(0, 3);
            } else if (g.type === 'cylinder') {
                // MuJoCo: size=[radius] or size=[radius, halfHeight]
                g.radius = g.size[0];
                if (g.size.length > 1) g.halfHeight = g.size[1];
                if (ge.getAttribute('fromto')) {
                    g.fromto = _parseVec(ge.getAttribute('fromto'), 6);
                    const ft = g.fromto;
                    g.halfHeight = Math.sqrt((ft[3]-ft[0])**2+(ft[4]-ft[1])**2+(ft[5]-ft[2])**2) / 2;
                }
                if (!g.halfHeight) g.halfHeight = 0.1;
            }
            g.density = parseFloat(ge.getAttribute('density') || '1000');
            geoms.push(g);
        }

        // Compute mass, COM, and inertia from geom shapes (matching MuJoCo).
        // MuJoCo: capsule = cylinder + 2 hemisphere caps, inertia in world-aligned frame.
        const PI = Math.PI;

        // Helper: compute geom mass and center position in body frame
        function geomMassAndCenter(g) {
            let volume = 0, center = g.pos.slice();
            if (g.type === 'sphere') {
                volume = (4/3) * PI * g.radius ** 3;
            } else if (g.type === 'capsule' && g.fromto) {
                const ft = g.fromto, r = g.radius;
                const halfH = Math.sqrt((ft[3]-ft[0])**2+(ft[4]-ft[1])**2+(ft[5]-ft[2])**2) / 2;
                volume = PI * r*r * (2*halfH) + (4/3) * PI * r**3;
                center = [(ft[0]+ft[3])/2, (ft[1]+ft[4])/2, (ft[2]+ft[5])/2]; // midpoint
            } else if (g.type === 'capsule') {
                const r = g.radius, hh = g.size.length > 1 ? g.size[1] : 0.1;
                volume = PI * r*r * (2*hh) + (4/3) * PI * r**3;
            } else if (g.type === 'box') {
                const he = g.halfExtents || g.size.slice(0,3);
                volume = 8 * he[0] * he[1] * he[2];
            } else if (g.type === 'cylinder') {
                const r = g.radius, hh = g.halfHeight || 0.1;
                volume = PI * r*r * (2*hh);
                if (g.fromto) {
                    const ft = g.fromto;
                    center = [(ft[0]+ft[3])/2, (ft[1]+ft[4])/2, (ft[2]+ft[5])/2];
                }
            }
            return { mass: volume * g.density, center, volume };
        }

        // Helper: compute capsule inertia tensor in WORLD-ALIGNED frame
        // MuJoCo computes capsule inertia along its axis then rotates to world frame
        function capsuleInertia(g) {
            const ft = g.fromto, r = g.radius;
            const dx = ft[3]-ft[0], dy = ft[4]-ft[1], dz = ft[5]-ft[2];
            const L = Math.sqrt(dx*dx+dy*dy+dz*dz);
            const halfH = L / 2;

            // Cylinder part
            const cylVol = PI * r*r * L;
            const cylM = cylVol * g.density;
            // Sphere caps (two hemispheres = one full sphere)
            const sphVol = (4/3) * PI * r**3;
            const sphM = sphVol * g.density;
            const totalM = cylM + sphM;

            // Inertia about capsule axis (local Z) and perpendicular
            // Cylinder: Iaxial = m*r^2/2, Iperp = m*(3r^2+L^2)/12
            const cylIaxial = cylM * r*r / 2;
            const cylIperp = cylM * (3*r*r + L*L) / 12;
            // Sphere: Iaxial = 2mr^2/5, Iperp = 2mr^2/5 + m*(3/8*r + halfH)^2 (parallel axis)
            const sphIaxial = 2 * sphM * r*r / 5;
            const sphIperp = sphIaxial + sphM * (3*r/8 + halfH)**2;

            const Iaxial = cylIaxial + sphIaxial;
            const Iperp = cylIperp + sphIperp;

            // Rotate from capsule-local to world-aligned frame
            // Capsule axis direction
            if (L < 1e-10) return [Iperp, Iperp, Iperp]; // degenerate
            const ax = [dx/L, dy/L, dz/L];
            // For diagonal inertia in world frame: I_world_ii = Iperp + (Iaxial-Iperp)*ax_i^2
            // This is the diagonal of R * diag(Iperp,Iperp,Iaxial) * R^T
            return [
                Iperp + (Iaxial - Iperp) * ax[0]*ax[0],
                Iperp + (Iaxial - Iperp) * ax[1]*ax[1],
                Iperp + (Iaxial - Iperp) * ax[2]*ax[2],
            ];
        }

        let totalMass = 0;
        let com = [0, 0, 0];
        const geomData = geoms.map(g => {
            const { mass, center } = geomMassAndCenter(g);
            return { g, mass, center };
        });
        for (const { mass, center } of geomData) {
            com[0] += mass * center[0];
            com[1] += mass * center[1];
            com[2] += mass * center[2];
            totalMass += mass;
        }
        if (totalMass > 0) com = com.map(c => c / totalMass);

        let inertia = [0, 0, 0];
        for (const { g, mass, center } of geomData) {
            let Ii;
            if (g.type === 'sphere') {
                const I = (2/5) * mass * g.radius**2;
                Ii = [I, I, I];
            } else if (g.type === 'capsule' && g.fromto) {
                Ii = capsuleInertia(g);
            } else if (g.type === 'box') {
                const he = g.halfExtents || g.size.slice(0,3);
                const a=2*he[0], b=2*he[1], c=2*he[2];
                Ii = [mass*(b*b+c*c)/12, mass*(a*a+c*c)/12, mass*(a*a+b*b)/12];
            } else if (g.type === 'cylinder') {
                const r = g.radius, hh = g.halfHeight || 0.015;
                const Iaxial = mass * r*r / 2;
                const Iperp = mass * (3*r*r + (2*hh)**2) / 12;
                if (g.fromto) {
                    const ft = g.fromto, L = 2*hh;
                    const ddx = ft[3]-ft[0], ddy = ft[4]-ft[1], ddz = ft[5]-ft[2];
                    if (L > 1e-10) {
                        const ax = [ddx/L, ddy/L, ddz/L];
                        Ii = [Iperp + (Iaxial-Iperp)*ax[0]*ax[0],
                              Iperp + (Iaxial-Iperp)*ax[1]*ax[1],
                              Iperp + (Iaxial-Iperp)*ax[2]*ax[2]];
                    } else { Ii = [Iperp, Iperp, Iaxial]; }
                } else { Ii = [Iperp, Iperp, Iaxial]; }
            } else {
                Ii = [0, 0, 0];
            }
            // Parallel axis theorem: shift from geom center to body COM
            const dx = center[0]-com[0], dy = center[1]-com[1], dz = center[2]-com[2];
            inertia[0] += Ii[0] + mass*(dy*dy+dz*dz);
            inertia[1] += Ii[1] + mass*(dx*dx+dz*dz);
            inertia[2] += Ii[2] + mass*(dx*dx+dy*dy);
        }

        bodies.push({ name, parent: parentName, pos: worldPos, localPos, geoms,
                       mass: totalMass, inertia, com });

        // Joints
        const jointEls = [...el.querySelectorAll(':scope > joint')].filter(
            j => (j.getAttribute('type') || 'hinge') !== 'free' && j.tagName !== 'freejoint');

        if (fixedBodies.has(name)) {
            fixedJoints.push({ name: name+'_fixed', parent_body: parentName, child_body: name, localPos0: localPos });
        } else if (jointEls.length && parentName !== null) {
            const axesData = [], jointAxes = [];
            for (const je of jointEls) {
                const axis = _parseVec(je.getAttribute('axis') || '1 0 0', 3);
                jointAxes.push(axis);
                const rng = _parseVec(je.getAttribute('range') || '-3.14159 3.14159', 2);
                const jname = je.getAttribute('name');
                const act = actuatorMap[jname] || { gear: 100, maxForce: 100 };
                axesData.push({
                    name: jname, mjcf_axis: axis,
                    stiffness: parseFloat(je.getAttribute('stiffness') || '0'),
                    damping: parseFloat(je.getAttribute('damping') || '0'),
                    maxForce: act.maxForce,
                    range: rng,
                    armature: parseFloat(je.getAttribute('armature') || '0'),
                });
            }

            const { q, axisMap } = _computeJointFrame(jointAxes);
            const localRot = [q[3], q[0], q[1], q[2]]; // xyzw → wxyz

            joints.push({
                name: jointEls.length > 1 ? jointEls[0].getAttribute('name').replace(/_[^_]+$/, '') : jointEls[0].getAttribute('name'),
                parent_body: parentName, child_body: name,
                axes: axesData,
                axisMap: axisMap.slice(0, jointEls.length),
                localPos0: localPos,
                localRot,
                jointType: jointEls.length > 1 ? 'spherical' : 'revolute',
            });
        }

        for (const child of el.querySelectorAll(':scope > body'))
            processBody(child, name, worldPos);
    }

    const worldbody = root.querySelector('worldbody');
    const pelvis = worldbody.querySelector(':scope > body');
    const pelvisPos = _parseVec(pelvis.getAttribute('pos'), 3) || [0,0,0];
    processBody(pelvis, null, pelvisPos);

    // Build dofInfo from actuator order
    const dofInfo = [];
    for (const actName of actuatorOrder) {
        for (const jdata of joints) {
            for (let ai = 0; ai < jdata.axes.length; ai++) {
                if (jdata.axes[ai].name === actName) {
                    dofInfo.push({
                        joint_name: jdata.name, axis_name: actName,
                        physx_axis: jdata.axisMap[ai], child_body: jdata.child_body,
                    });
                    break;
                }
            }
        }
    }

    // Build FK data
    const bodyNames = bodies.map(b => b.name);
    const fk_parent_indices = bodies.map(b => b.parent === null ? -1 : bodyNames.indexOf(b.parent));
    const fk_local_translations = bodies.map(b => b.localPos);
    const fk_local_rotations = bodies.map(() => [0, 0, 0, 1]); // all identity for MJCF

    // Build kinematicJoints (for obs computation)
    const kinematicJoints = [];
    let dofIdx = 0;
    for (let bi = 1; bi < bodies.length; bi++) {
        const bname = bodies[bi].name;
        const jdata = joints.find(j => j.child_body === bname);
        const fjdata = fixedJoints.find(j => j.child_body === bname);
        if (jdata) {
            const nDofs = jdata.axes.length;
            kinematicJoints.push({
                name: jdata.name, child_body: bname,
                type: nDofs > 1 ? 'SPHERICAL' : 'HINGE',
                dof_idx: dofIdx, dof_dim: nDofs,
                axis: nDofs === 1 ? jdata.axes[0].mjcf_axis : undefined,
            });
            dofIdx += nDofs;
        } else if (fjdata) {
            kinematicJoints.push({ name: fjdata.name, child_body: bname, type: 'FIXED', dof_idx: dofIdx, dof_dim: 0 });
        } else {
            kinematicJoints.push({ name: bname, child_body: bname, type: 'FIXED', dof_idx: dofIdx, dof_dim: 0 });
        }
    }

    return {
        bodies, joints, fixedJoints, actuatorOrder, dofInfo,
        fk_parent_indices, fk_local_translations, fk_local_rotations,
        kinematicJoints,
        act_dim: actuatorOrder.length,
    };
}

// Export for use as ES module or global
if (typeof module !== 'undefined') module.exports = { parseMJCF };
