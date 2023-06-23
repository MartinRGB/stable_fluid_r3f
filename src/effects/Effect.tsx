import { OrbitControls, Plane,Circle, shaderMaterial } from "@react-three/drei"
import { Canvas, useThree, useFrame, createPortal } from "@react-three/fiber"
import { useControls } from "leva";
import {  useEffect, useMemo, useRef, useState } from "react";
import * as THREE from 'three'
import { Perf } from "r3f-perf";

const prefix_vertex = `
    varying vec2 vUv;
    varying vec3 v_pos;

`

const common_vertex_main = `
    void main()	{
        vUv = uv;
        v_pos = position;
        gl_Position = vec4(position, 1.);
    }
`

const prefix_frag = `
    #ifdef GL_ES
    precision highp float;
    #endif

    varying vec3 v_pos;
    varying vec2 vUv;

    uniform float time;
    uniform vec2 resolution;
    uniform vec2 mouse;
    #define iResolution resolution
    #define iTime time
    #define iMouse mouse

`

const face_vert=`
//attribute vec3 position;
uniform vec2 px;
uniform vec2 boundarySpace;
varying vec2 uv_2;
varying vec2 vUv;
varying vec3 v_pos;

precision highp float;

void main(){
    v_pos = position;
    vUv = uv;
    vec3 pos = position;
    vec2 scale = 1.0 - boundarySpace * 2.0;
    pos.xy = pos.xy * scale;
    uv_2 = vec2(0.5)+(pos.xy)*0.5;
    gl_Position = vec4(pos, 1.0);
}`

const basic_uniform = {
    resolution:[null,null],
    time:0,
    mouse:[null,null],
}

const renderedFBOTexture = (gl:THREE.WebGLRenderer,fboInput:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,scene:THREE.Scene,camera:THREE.Camera) =>{
    gl.setRenderTarget(fboInput);
    gl.render(scene,camera)
    gl.setRenderTarget(null)
    //return fboInput.texture;
} 


interface AdvectionSolveProgramProps{
    scene?:any,
    camera?:THREE.Camera,
    isBounce?:boolean,
    cellScale?:[number,number],
    fboSize?:[number,number],
    dt?:number,
    src?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dst?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    isBFECC?:boolean,
}

const AdvectionSolveProgram = ({scene,camera,isBounce,cellScale,fboSize,dt,src,dst,isBFECC}:AdvectionSolveProgramProps) =>{
    const lineMatRef = useRef<THREE.ShaderMaterial | null>(null)
    const faceMatRef = useRef<THREE.ShaderMaterial | null>(null)

    const line_vert = `
    //attribute vec3 position;
    varying vec2 uv_2;
    varying vec2 vUv;
    varying vec3 v_pos;
    uniform vec2 px;
    
    precision highp float;
    
    void main(){
        v_pos = position;
        vUv = uv;
        vec3 pos = position;
        uv_2 = 0.5 + pos.xy * 0.5;
        vec2 n = sign(pos.xy);
        pos.xy = abs(pos.xy) - px * 1.0;
        pos.xy *= n;
        gl_Position = vec4(pos, 1.0);
    }`
    
    
    const advection_frag=`
    precision highp float;
    uniform sampler2D velocity;
    uniform float dt;
    uniform bool isBFECC;
    // uniform float uvScale;
    uniform vec2 fboSize;
    uniform vec2 px;
    varying vec2 uv_2;
    
    
    void main(){
        vec2 ratio = max(fboSize.x, fboSize.y) / fboSize;
    
        if(isBFECC == false){
            vec2 vel = texture2D(velocity, uv_2).xy;
            vec2 uv2 = uv_2 - vel * dt * ratio;
            vec2 newVel = texture2D(velocity, uv2).xy;
            gl_FragColor = vec4(newVel, 0.0, 0.0);
        } else {
            vec2 spot_new = uv_2;
            vec2 vel_old = texture2D(velocity, uv_2).xy;
            // back trace
            vec2 spot_old = spot_new - vel_old * dt * ratio;
            vec2 vel_new1 = texture2D(velocity, spot_old).xy;
    
            // forward trace
            vec2 spot_new2 = spot_old + vel_new1 * dt * ratio;
            
            vec2 error = spot_new2 - spot_new;
    
            vec2 spot_new3 = spot_new - error / 2.0;
            vec2 vel_2 = texture2D(velocity, spot_new3).xy;
    
            // back trace 2
            vec2 spot_old2 = spot_new3 - vel_2 * dt * ratio;
            // gl_FragColor = vec4(spot_old2, 0.0, 0.0);
            vec2 newVel2 = texture2D(velocity, spot_old2).xy; 
            gl_FragColor = vec4(newVel2, 0.0, 0.0);
        }


    }`
    
    

    // # Advection Line Geometry
    const vertices_boundary = useMemo(() =>{
        return new Float32Array([
            // left
            -1, -1, 0,
            -1, 1, 0,
        
            // top
            -1, 1, 0,
            1, 1, 0,
        
            // right
            1, 1, 0,
            1, -1, 0,
        
            // bottom
            1, -1, 0,
            -1, -1, 0
        ]
        //.map(x=>x*2.)
        );
    },[])

    //https://stackoverflow.com/questions/67555786/custom-buffergeometry-in-react-three-fiber

    useFrame(({clock,gl})=>{
        if(faceMatRef.current && lineMatRef.current && src){
            faceMatRef.current.uniforms.velocity.value = src.texture;
            lineMatRef.current.uniforms.velocity.value = src.texture;


            // render the scene to the render target as output
            if(dst && camera){
                renderedFBOTexture(gl,dst,scene,camera);
            }


        }
    })

    return(
    <>
        {createPortal(<>
            <Plane args={[2,2]}>
                    <shaderMaterial
                        ref={faceMatRef}
                        uniforms={
                            {
                                boundarySpace: { value: cellScale?cellScale:[null,null] },
                                px: { value:cellScale?cellScale:[null,null] },
                                fboSize: { value:fboSize?fboSize:[null,null] },
                                velocity: { value:src?src.texture:null },
                                dt: { value:dt?dt:null },
                                isBFECC:{ value:isBFECC?isBFECC:null },
                            }
                        }
                        vertexShader={face_vert}
                        fragmentShader={advection_frag}

                    ></shaderMaterial>
            </Plane>
            {/* {isBounce? */}
            <line>
                <bufferGeometry attach="geometry" >
                <bufferAttribute
                attach={'attributes-position'}
                array={vertices_boundary}
                count={vertices_boundary.length / 3}
                itemSize={3}
                />
                </bufferGeometry>
                <shaderMaterial
                        ref={lineMatRef}
                        uniforms={
                            {
                                boundarySpace: { value: cellScale?cellScale:[null,null] },
                                px: { value:cellScale?cellScale:[null,null] },
                                fboSize: { value:fboSize?fboSize:[null,null] },
                                velocity: { value:src?src.texture:null },
                                dt: { value:dt?dt:null },
                                isBFECC:{ value:isBFECC?isBFECC:null },
                            }
                        }
                        vertexShader={line_vert}
                        fragmentShader={advection_frag}

                    ></shaderMaterial>
            </line>
            {/* :<></>} */}
        </>,scene)}
    </>)
};
AdvectionSolveProgram.displayName = 'AdvectionSolveProgram'

interface ExternalForceSolveProgramProps{
    scene?:any,
    camera?:THREE.Camera,
    cellScale?:[number,number],
    scale?:[number,number],
    mouse_force?:number,
    dst?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
}

const ExternalForceProgram = ({scene,camera,cellScale,scale,mouse_force,dst}:ExternalForceSolveProgramProps) =>{

    let mouse_vert=`
    precision highp float;

    uniform vec2 center;
    uniform vec2 scale;
    uniform vec2 px;
    varying vec2 vUv;

    void main(){
        vec2 pos = position.xy * scale * 2.0 * px + center;
        vUv = uv;
        gl_Position = vec4(pos, 0.0, 1.0);
    }`

    let externalForce_frag=`precision highp float;
    uniform vec2 force;
    uniform vec2 center;
    uniform vec2 scale;
    uniform float mouse_force;
    uniform float time;
    uniform vec2 px;
    varying vec2 vUv;

    #define outFlowStrength 0.2
    #define PI 3.1415926535897932384626433832795

    void main(){
        vec2 circle = (vUv - 0.5) * 2.0;
        float d = 1.0-min(length(circle), 1.0);
        d *= d;
        gl_FragColor = vec4(force * d, 0, 1);
        //float timeClock = fract(time) * 2.;
        //gl_FragColor = vec4(sin(PI * timeClock) * 0.3, cos(PI*timeClock) * 0.3, 0, 1);
    }
    `

    useEffect(()=>{
        document.body.addEventListener( 'mousemove',onDocumentMouseMove.bind(this), false );
        document.body.addEventListener( 'touchstart', onDocumentTouchStart.bind(this), false );
        document.body.addEventListener( 'touchmove', onDocumentTouchMove.bind(this), false );
    },[])

    let [mouseMoved,coords,coords_old,diff] = useMemo(()=>{
        return [false,new THREE.Vector2(),new THREE.Vector2(),new THREE.Vector2()]
    },[])

    const updateCoords = ( x:number, y:number ) => {

        coords.set( ( x / window.innerWidth ) * 2 - 1, - ( y / window.innerHeight ) * 2 + 1 );
        mouseMoved = true;
        setTimeout(() => {
            mouseMoved = false;
        }, 100);
        
    }

    const onDocumentMouseMove = ( event:MouseEvent )  => {
        updateCoords( event.clientX, event.clientY );
    }
    const onDocumentTouchStart = ( event:TouchEvent )  => {
        if ( event.touches.length === 1 ) {
            // event.preventDefault();
            updateCoords( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY );
        }
    }
    const onDocumentTouchMove = ( event:TouchEvent )  => {
        if ( event.touches.length === 1 ) {
            // event.preventDefault();
            updateCoords( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY );
        }
    }

    const updateMouse = () => {
        diff.subVectors(coords, coords_old);
        coords_old.copy(coords);

        if(coords_old.x === 0 && coords_old.y === 0) diff.set(0, 0);
    }

    useFrame(({clock,gl})=>{
        updateMouse()
        if(mouseMatRef.current){
            mouseMatRef.current.uniforms.time.value = clock.getElapsedTime() ;
        }
        if(mouse_force && cellScale && mouseMatRef.current){
            const forceX = diff.x / 2 * mouse_force;
            const forceY = diff.y / 2 * mouse_force;
    
            const cursor_size = scale?scale[0]:0;
    
            const cursorSizeX = cursor_size * cellScale[0];
            const cursorSizeY = cursor_size * cellScale[1];
    
            const centerX = Math.min(Math.max(coords.x, -1 + cursorSizeX + cellScale[0] * 2), 1 - cursorSizeX - cellScale[0] * 2);
            const centerY = Math.min(Math.max(coords.y, -1 + cursorSizeY + cellScale[1] * 2), 1 - cursorSizeY - cellScale[1] * 2);
            
            mouseMatRef.current.uniforms.force.value = [forceX,forceY];
            mouseMatRef.current.uniforms.center.value = [centerX,centerY];
            mouseMatRef.current.uniforms.scale.value = [cursor_size,cursor_size];

        }
        // render the scene to the render target as output
        if(dst && camera)
            renderedFBOTexture(gl,dst,scene,camera);
    })

    const mouseMatRef = useRef<any>();

    return(
    <>
        {createPortal(<>
                <Circle args={[0.25,32]}>
                    <shaderMaterial
                        ref={mouseMatRef}
                        blending={THREE.AdditiveBlending}
                        uniforms={
                            {
                                px: { value:cellScale?cellScale:[null,null] },
                                force: { value:[0.,0.] },
                                center: { value:[0.,0.] },
                                scale: { value:scale?scale:[null,null] },
                                time:{value:0.}
                            }
                        }
                        vertexShader={mouse_vert} //mouse_vert
                        fragmentShader={externalForce_frag}>
                    
                    </shaderMaterial>
                </Circle>
            </>,scene)
        }
    </>
    )
        
};
ExternalForceProgram.displayName = 'ExternalForceProgram'

interface FieldForceProgramProps{
    scene?:any,
    camera?:THREE.Camera,
    dst?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
}

const FieldForceProgram = ({scene,camera,dst}:FieldForceProgramProps) =>{


    const field_frag_prefix = `
    #define MAPRES vec2(64,64);
    #define PI 3.1415926535897932384626433832795
    #define HALFPI 1.57079632679

    // Hash Part
    #define ITERATIONS 4


    // *** Change these to suit your range of random numbers..

    // *** Use this for integer stepped ranges, ie Value-Noise/Perlin noise functions.
    #define HASHSCALE1 .1031
    #define HASHSCALE3 vec3(.1031, .1030, .0973)
    #define HASHSCALE4 vec4(.1031, .1030, .0973, .1099)

    // For smaller input rangers like audio tick or 0-1 UVs use these...
    //#define HASHSCALE1 443.8975
    //#define HASHSCALE3 vec3(443.897, 441.423, 437.195)
    //#define HASHSCALE4 vec3(443.897, 441.423, 437.195, 444.129)



    //----------------------------------------------------------------------------------------
    //  1 out, 1 in...
    float hash11(float p)
    {
        vec3 p3  = fract(vec3(p) * HASHSCALE1);
        p3 += dot(p3, p3.yzx + 19.19);
        return fract((p3.x + p3.y) * p3.z);
    }

    //----------------------------------------------------------------------------------------
    //  1 out, 2 in...
    float hash12(vec2 p)
    {
        vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
        p3 += dot(p3, p3.yzx + 19.19);
        return fract((p3.x + p3.y) * p3.z);
    }

    //----------------------------------------------------------------------------------------
    //  1 out, 3 in...
    float hash13(vec3 p3)
    {
        p3  = fract(p3 * HASHSCALE1);
        p3 += dot(p3, p3.yzx + 19.19);
        return fract((p3.x + p3.y) * p3.z);
    }

    //----------------------------------------------------------------------------------------
    //  2 out, 1 in...
    vec2 hash21(float p)
    {
        vec3 p3 = fract(vec3(p) * HASHSCALE3);
        p3 += dot(p3, p3.yzx + 19.19);
        return fract((p3.xx+p3.yz)*p3.zy);

    }

    //----------------------------------------------------------------------------------------
    ///  2 out, 2 in...
    vec2 hash22(vec2 p)
    {
        vec3 p3 = fract(vec3(p.xyx) * HASHSCALE3);
        p3 += dot(p3, p3.yzx+19.19);
        return fract((p3.xx+p3.yz)*p3.zy);

    }

    //----------------------------------------------------------------------------------------
    ///  2 out, 3 in...
    vec2 hash23(vec3 p3)
    {
            p3 = fract(p3 * HASHSCALE3);
        p3 += dot(p3, p3.yzx+19.19);
        return fract((p3.xx+p3.yz)*p3.zy);
    }

    //----------------------------------------------------------------------------------------
    //  3 out, 1 in...
    vec3 hash31(float p)
    {
    vec3 p3 = fract(vec3(p) * HASHSCALE3);
    p3 += dot(p3, p3.yzx+19.19);
    return fract((p3.xxy+p3.yzz)*p3.zyx); 
    }


    //----------------------------------------------------------------------------------------
    ///  3 out, 2 in...
    vec3 hash32(vec2 p)
    {
        vec3 p3 = fract(vec3(p.xyx) * HASHSCALE3);
        p3 += dot(p3, p3.yxz+19.19);
        return fract((p3.xxy+p3.yzz)*p3.zyx);
    }

    //----------------------------------------------------------------------------------------
    ///  3 out, 3 in...
    vec3 hash33(vec3 p3)
    {
        p3 = fract(p3 * HASHSCALE3);
        p3 += dot(p3, p3.yxz+19.19);
        return fract((p3.xxy + p3.yxx)*p3.zyx);

    }

    //----------------------------------------------------------------------------------------
    // 4 out, 1 in...
    vec4 hash41(float p)
    {
        vec4 p4 = fract(vec4(p) * HASHSCALE4);
        p4 += dot(p4, p4.wzxy+19.19);
        return fract((p4.xxyz+p4.yzzw)*p4.zywx);
        
    }

    //----------------------------------------------------------------------------------------
    // 4 out, 2 in...
    vec4 hash42(vec2 p)
    {
        vec4 p4 = fract(vec4(p.xyxy) * HASHSCALE4);
        p4 += dot(p4, p4.wzxy+19.19);
        return fract((p4.xxyz+p4.yzzw)*p4.zywx);

    }

    //----------------------------------------------------------------------------------------
    // 4 out, 3 in...
    vec4 hash43(vec3 p)
    {
        vec4 p4 = fract(vec4(p.xyzx)  * HASHSCALE4);
        p4 += dot(p4, p4.wzxy+19.19);
        return fract((p4.xxyz+p4.yzzw)*p4.zywx);
    }

    //----------------------------------------------------------------------------------------
    // 4 out, 4 in...
    vec4 hash44(vec4 p4)
    {
            p4 = fract(p4  * HASHSCALE4);
        p4 += dot(p4, p4.wzxy+19.19);
        return fract((p4.xxyz+p4.yzzw)*p4.zywx);
    }
    `

    const field_frag=`
    uniform float time;
    uniform vec2 resolution;
    uniform sampler2D frame_texure;
    #define iChannel0 frame_texure
    #define iResolution resolution
    #define iTime time

    vec2 get_velocity(in vec2 p){
        p = p * vec2(iResolution.x,iResolution.y)/MAPRES;
        vec2 v = vec2(0.);
        //==================== write your code here ==================
        // v.x=sin(p.x);
        // v.y=cos(p.y);
        v.x=1.0;
        v.y=0.0;
        //============================================================
        return v;
    }

    vec2 field(vec2 fragCoord) {
        //I.   generate a staggered grid,
        //     caculation in simulation will use centred difference
        //     centred difference is  a more accurate approximation
        //II.  readStoredPosition, get particles' position from texture => p
        //III. generate random position => p = p + noise
        //IV.  get the velocity
        //V.   newPosition = position + velcotiy;
        
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                //this loop function generate a grid
                //get 9 point from center of fragCoord coordinate
                vec2 uv = (fragCoord + vec2(i,j)) / iResolution.xy; 
                vec2 p = texture(iChannel0, fract(uv)).xy;
                if(p == vec2(0)) {
                    // if there is noise point in this coordniate,the particle will exist,or return vec2(0.);
                    if (hash13(vec3(fragCoord + vec2(i,j), iTime)) > 1e-4) continue;
                    // in fact,the random hash value did not affect the final efx.
                    p = fragCoord + vec2(i,j) + hash21(float(iTime)) - 0.5; // add particle
                    
                } else if (hash13(vec3(fragCoord + vec2(i,j), iTime)) < 8e-3) {
                    continue; // remove particle
                }
                vec2 v = get_velocity((uv*2. - vec2(0.5,0.5*iResolution.x/iResolution.y)));
                p = p + v; //newPosition
                p.x = mod(p.x, iResolution.x);

                // this means,control the pariticle in the grid
                if(abs(p.x - fragCoord.x) < 0.5 && abs(p.y - fragCoord.y) < 0.5)
                    return p;
            }
        }
        
        return vec2(0.);
    }
    `

    let field_force_frag=`precision highp float;
    uniform vec2 px;
    varying vec2 vUv;

    void main(){
        vec2 circle = (vUv - 0.5) * 2.0;
        float d = 1.0-min(length(circle), 1.0);
        d *= d;
        //gl_FragColor = vec4(force * d, 0, 1);
        vec2 fieldPos = field(gl_FragCoord.xy);
        gl_FragColor = vec4(fieldPos,0.,1.);
    }
    `



    useFrame(({clock,gl})=>{
        if(fieldForceMatRef.current && dst){
            fieldForceMatRef.current.uniforms.time.value = clock.getElapsedTime();
            fieldForceMatRef.current.uniforms.frame_texture.value = dst.texture;
        }

        if(dst && camera)
            renderedFBOTexture(gl,dst,scene,camera);

    })

    const fieldForceMatRef = useRef<any>();
    const {size} = useThree();

    return(
    <>
        {createPortal(<>
                <Plane args={[2,2]}>
                    <shaderMaterial
                        ref={fieldForceMatRef}
                        blending={THREE.AdditiveBlending}
                        uniforms={
                            {
                                time: { value:0 },
                                resolution: { value:[size.width,size.height] },
                                frame_texture:{value:null}
                            }
                        }
                        vertexShader={face_vert} //mouse_vert
                        fragmentShader={field_frag_prefix + field_frag +  field_force_frag}>
                    
                    </shaderMaterial>
                </Plane>
            </>,scene)
        }
    </>
    )
        
};
FieldForceProgram.displayName = 'FieldForceProgramProgram'

interface ViscousSolveProgramProps{
    scene?:any,
    camera?:THREE.Camera,
    iterations_viscous?:number,
    cellScale?:[number,number],
    boundarySpace?:[number,number],
    viscous?:number,
    src?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dst?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dst_?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dt?:number
}

const ViscousSolveProgram = ({scene,camera,iterations_viscous,cellScale,boundarySpace,viscous,src,dst,dst_,dt}:ViscousSolveProgramProps) =>{

    let viscous_frag=`precision highp float;
    uniform sampler2D velocity;
    uniform sampler2D velocity_new;
    uniform float v;
    uniform vec2 px;
    uniform float dt;

    varying vec2 uv_2;
    
    void main(){
        // poisson equation
        vec2 old = texture2D(velocity, uv_2).xy;
        vec2 new0 = texture2D(velocity_new, uv_2 + vec2(px.x * 2.0, 0)).xy;
        vec2 new1 = texture2D(velocity_new, uv_2 - vec2(px.x * 2.0, 0)).xy;
        vec2 new2 = texture2D(velocity_new, uv_2 + vec2(0, px.y * 2.0)).xy;
        vec2 new3 = texture2D(velocity_new, uv_2 - vec2(0, px.y * 2.0)).xy;
    
        vec2 new = 4.0 * old + v * dt * (new0 + new1 + new2 + new3);
        new /= 4.0 * (1.0 + v * dt);
        
        gl_FragColor = vec4(new, 0.0, 0.0);
    }
    `

    const viscousMatRef = useRef<THREE.ShaderMaterial | null>(null)

    useFrame(({clock,gl})=>{
        if(viscousMatRef.current && viscous){
            viscousMatRef.current.uniforms.v.value = viscous;
        }


        if(iterations_viscous && dst && dst_){

            // for(var i = 0; i < iterations_viscous; i++){
            //     let isEven = i  % 2 ==0;

            //     if(viscousMatRef.current){
            //         viscousMatRef.current.uniforms.velocity_new.value = isEven?dst_.texture:dst.texture;
            //     }
                
            //     if(viscousMatRef.current && dt){
            //         viscousMatRef.current.uniforms.dt.value = dt;
            //     }

            //     // render the scene to the render target as output
            //     if(camera)
            //         renderedFBOTexture(gl,isEven?dst:dst_,scene,camera);
            // }

            // ### Swap Buffer Method ###
            for(var i = 0; i < iterations_viscous; i++){
                // render the scene to the render target as output
                if(camera && dst)
                    renderedFBOTexture(gl,dst,scene,camera)
                    
                if(viscousMatRef.current && dst){
                    viscousMatRef.current.uniforms.velocity_new.value = dst.texture;
                }

                if(viscousMatRef.current && dt){
                    viscousMatRef.current.uniforms.dt.value = dt;
                }

                // swap buffer
                let t1 = dst;
                dst = dst_;
                dst_ = t1;
            
            }

        }

        

    })

    return(
        <>
        {createPortal(<>
                <Plane args={[2,2]}>
                    <shaderMaterial
                        ref={viscousMatRef}
                        uniforms={
                            {
                                boundarySpace: { value: boundarySpace?boundarySpace:[null,null] },
                                velocity: { value: src?src.texture:null },
                                velocity_new: { value: dst_?dst_.texture:null },
                                v:{ value: viscous?viscous:null },
                                px: { value: cellScale?cellScale:[null,null] },
                                dt: { value: dt?dt:null },
        
                            }
                        }
                        vertexShader={face_vert}
                        fragmentShader={viscous_frag} //viscous_frag
    
                    ></shaderMaterial>
                </Plane>
            </>,scene)
        }
        </>
    )
};

ViscousSolveProgram.displayName = 'ViscousSolveProgram';

interface DivergenceSolveProgramProps{
    scene?:any,
    camera?:THREE.Camera,
    cellScale?:[number,number],
    boundarySpace?:[number,number],
    src?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dst?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    vel?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dt?:number
}

const DivergenceSolveProgram = ({scene,camera,cellScale,boundarySpace,src,dst,vel,dt}:DivergenceSolveProgramProps) => {
    
    const divergence_frag=`precision highp float;
    uniform sampler2D velocity;
    uniform float dt;
    uniform vec2 px;
    
    varying vec2 uv_2;
    
    void main(){
        float x0 = texture2D(velocity, uv_2-vec2(px.x, 0)).x;
        float x1 = texture2D(velocity, uv_2+vec2(px.x, 0)).x;
        float y0 = texture2D(velocity, uv_2-vec2(0, px.y)).y;
        float y1 = texture2D(velocity, uv_2+vec2(0, px.y)).y;
        float divergence = (x1-x0 + y1-y0) / 2.0;
    
        gl_FragColor = vec4(divergence / dt);
        
    }`

    useFrame(({clock,gl})=>{
        if(divergenceMatRef.current && vel){
            divergenceMatRef.current.uniforms.velocity.value = vel.texture;
        }
        
        // render the scene to the render target as output
        if(dst && camera)
            renderedFBOTexture(gl,dst,scene,camera)

    })
    
    const divergenceMatRef = useRef<THREE.ShaderMaterial | null>(null)


    return(
        <>
            {createPortal(<>
            
            <Plane args={[2,2]}>
                <shaderMaterial
                        ref={divergenceMatRef}
                        uniforms={
                            {
                                boundarySpace: { value: boundarySpace?boundarySpace:[null,null] },
                                velocity: { value: src?src.texture:null },
                                px: { value: cellScale?cellScale:[null,null] },
                                dt: { value: dt?dt:null },
                                time:{ value:null},
                            }
                        }
                        vertexShader={face_vert}
                        fragmentShader={divergence_frag}
    
                    ></shaderMaterial>
            </Plane>
            </>,scene)}
        </>
    )
};

DivergenceSolveProgram.displayName = 'DivergenceSolveProgram';

interface PoissonSolveProgramProps{
    scene?:any,
    camera?:THREE.Camera,
    cellScale?:[number,number],
    boundarySpace?:[number,number],
    iterations_poisson?:number,
    src?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dst?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dst_?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
}

const PoissonSolveProgram = ({scene,camera,cellScale,boundarySpace,iterations_poisson,src,dst,dst_}:PoissonSolveProgramProps) => {
    
    const poisson_frag=`precision highp float;
    uniform sampler2D pressure;
    uniform sampler2D divergence;
    uniform vec2 px;
    varying vec2 uv_2;
    varying vec2 vUv;
    

    void main(){    
        // poisson equation
        float p0 = texture2D(pressure, uv_2+vec2(px.x * 2.0,  0)).r;
        float p1 = texture2D(pressure, uv_2-vec2(px.x * 2.0, 0)).r;
        float p2 = texture2D(pressure, uv_2+vec2(0, px.y * 2.0 )).r;
        float p3 = texture2D(pressure, uv_2-vec2(0, px.y * 2.0 )).r;
        float div = texture2D(divergence, vUv).r;
        
        float newP = (p0 + p1 + p2 + p3) / 4.0 - div;

        gl_FragColor = vec4(newP);
    }`

    useFrame(({clock,gl})=>{
        // bugs in renderer

        if(iterations_poisson && dst && dst_){
            
            // for(var i = 0; i < iterations_poisson; i++){

            //     let isEven = i  % 2 ==0;


            //     if(poissonMatRef.current){
            //         poissonMatRef.current.uniforms.pressure.value = isEven?dst_.texture:dst.texture;
            //     }
                

            //     // render the scene to the render target as output
            //     if(camera)
            //         renderedFBOTexture(gl,isEven?dst:dst_,scene,camera)
                
            // }   

            // ### Swap Buffer Method ###
            for(var i = 0; i < iterations_poisson; i++){
                // render the scene to the render target as output
                
                if(camera && dst)
                    renderedFBOTexture(gl,dst,scene,camera)
                    
                if(poissonMatRef.current && dst){
                    poissonMatRef.current.uniforms.pressure.value = dst.texture;
                }

                // swap buffer
                let t1 = dst;
                dst = dst_;
                dst_ = t1;
            
            }

        }
        
    })
    
    const poissonMatRef = useRef<THREE.ShaderMaterial | null>(null)
    const {size} = useThree();

    return(
        <>
        {createPortal(<>
            <Plane args={[2,2]}>
                <shaderMaterial
                        ref={poissonMatRef}
                        uniforms={
                            {
                                boundarySpace: { value: boundarySpace?boundarySpace:[null,null] },
                                pressure: { value: dst_?dst_.texture:null },
                                divergence: { value: src?src.texture:null },
                                px: { value: cellScale?cellScale:[null,null] },
                                time:{ value:null },
                                resolution:{ value:[size.width,size.height]}
                            }
                        }
                        vertexShader={face_vert}
                        fragmentShader={poisson_frag}
    
                    ></shaderMaterial>
            </Plane>
            </>,scene)}
        </>
    )
};

PoissonSolveProgram.displayName = 'PoissonSolveProgram';

interface PressureSolveProgramProps{
    scene?:any,
    camera?:THREE.Camera,
    cellScale?:[number,number],
    boundarySpace?:[number,number],
    src_p?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    src_v?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    src_update_p?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    src_update_v?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dst?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
    dt?:number
}

const PressureSolveProgram = ({scene,camera,cellScale,boundarySpace,src_p,src_v,src_update_p,src_update_v,dst,dt}:PressureSolveProgramProps) => {
    
    const pressure_frag=`precision highp float;
    uniform sampler2D pressure;
    uniform sampler2D velocity;
    uniform vec2 px;
    uniform float dt;
    varying vec2 uv_2;
    varying vec2 vUv;
    
    void main(){
        float step = 1.0;
    
        float p0 = texture2D(pressure, uv_2+vec2(px.x * step, 0)).r;
        float p1 = texture2D(pressure, uv_2-vec2(px.x * step, 0)).r;
        float p2 = texture2D(pressure, uv_2+vec2(0, px.y * step)).r;
        float p3 = texture2D(pressure, uv_2-vec2(0, px.y * step)).r;
    
        vec2 v = texture2D(velocity, uv_2).xy;
        vec2 gradP = vec2(p0 - p1, p2 - p3) * 0.5;
        v = v - gradP * dt;
        vec4 press_tex = texture2D(pressure,uv_2);
        vec4 vel_tex = texture2D(velocity,uv_2);
        gl_FragColor = vec4(v, 0.0, 1.0);
        //gl_FragColor = vel_tex;
    }`


    useFrame(({clock,gl})=>{
        if(pressureMatRef.current && src_update_v){
            pressureMatRef.current.uniforms.velocity.value = src_update_v.texture;
        }
        if(pressureMatRef.current && src_update_p){
            pressureMatRef.current.uniforms.pressure.value = src_update_p.texture;
        }

        // render the scene to the render target as output
        if(dst && camera)
            renderedFBOTexture(gl,dst,scene,camera);

    })


    
    const pressureMatRef = useRef<THREE.ShaderMaterial | null>(null)


    return(
        <>
            {createPortal(<>
            <Plane args={[2,2]}>
                <shaderMaterial
                        ref={pressureMatRef}
                        uniforms={
                            {
                                boundarySpace: { value: boundarySpace?boundarySpace:[null,null] },
                                pressure: { value: src_p?src_p.texture:null },
                                velocity: { value: src_v?src_v.texture:null },
                                px: { value: cellScale?cellScale:[null,null] },
                                dt: { value: dt?dt:null },
                            }
                        }
                        vertexShader={face_vert}
                        fragmentShader={pressure_frag}
    
                    ></shaderMaterial>
            </Plane>
            </>,scene)}

        </>
    )
};

PressureSolveProgram.displayName = 'PressureSolveProgram';



interface ColorProgramProps{
    src?:THREE.WebGLRenderTarget | THREE.WebGLMultipleRenderTargets | null,
}

const ColorProgram = ({src}:ColorProgramProps) => {
    const color_frag=`precision highp float;
    uniform sampler2D velocity;
    varying vec2 uv_2;
    
    void main(){
        vec4 texCol = texture2D(velocity, uv_2);
        vec2 vel = texture2D(velocity, uv_2).xy;
        float len = length(vel);
        vel = vel * 0.5 + 0.5;
        
        vec3 color = vec3(vel.x, vel.y, 1.0);
        color = mix(vec3(1.0), color, len);
    
        gl_FragColor = vec4(color,  1.0);
        //gl_FragColor = vec4(1. - color,  1.0);
    }`

    return(
        <>
                <Plane args={[2,2]}>
                    <shaderMaterial
                        uniforms={
                            {
                                velocity:{value:src?src.texture:null},
                                boundarySpace:{value:new THREE.Vector2},
                            }
                        }
                        vertexShader={face_vert}
                        fragmentShader={color_frag}
                    >

                    </shaderMaterial>
                </Plane>
        </>
    )
};

ColorProgram.displayName = 'ColorProgram';

const FluidSimulation = () =>{

    const {size,gl,camera} = useThree()



    // ### Parameters

    const iterations_poisson = 32;
    const iterations_viscous = 32;
    const resolution = 0.4;
    const mouse_force = 20;
    const cursor_size = 100;
    const viscous = 30;
    const isBounce = true;
    const dt = 0.014;
    const isViscous = false;
    const isBFECC = true;

    const [screenWidth,setScreenWidth] = useState(size.width );
    const [screenHeight,setScreenHeight] = useState(size.height );

    // # Scene
    const [advectionSolveScene,forceSolveScene,viscousSolveScene,divergenceSolveScene,poissonSolveScene,pressureSolveScene,

    ] = useMemo(()=>{
        return [new THREE.Scene(),new THREE.Scene(),new THREE.Scene(),new THREE.Scene(),new THREE.Scene(),new THREE.Scene(),

        ]
    },[])

    // # FBO
    let FBOSettings = { format: THREE.RGBAFormat,minFilter: THREE.LinearFilter,magFilter: THREE.LinearFilter,type: THREE.FloatType,}
    
    let [fbo_vel_0,fbo_vel_1,fbo_vel_viscous_0,fbo_vel_viscous_1,fbo_div,fbo_pressure_0,fbo_pressure_1,

    ] = useMemo(()=>{
        return [
            new THREE.WebGLRenderTarget(screenWidth*resolution,screenHeight*resolution,FBOSettings),
            new THREE.WebGLRenderTarget(screenWidth*resolution,screenHeight*resolution,FBOSettings),
            new THREE.WebGLRenderTarget(screenWidth*resolution,screenHeight*resolution,FBOSettings),
            new THREE.WebGLRenderTarget(screenWidth*resolution,screenHeight*resolution,FBOSettings),
            new THREE.WebGLRenderTarget(screenWidth*resolution,screenHeight*resolution,FBOSettings),
            new THREE.WebGLRenderTarget(screenWidth*resolution,screenHeight*resolution,FBOSettings),
            new THREE.WebGLRenderTarget(screenWidth*resolution,screenHeight*resolution,FBOSettings),

        ]
    },[])


    // // # Window Resize Function
    // const resizeFuns = (width:number,height:number) =>{
    //     setScreenWidth(width*resSize)
    //     setScreenHeight(height*resSize)
    //     //setMatrialUniforms(finalMat.current,{'resolution':[width,height]})
    // }

    // useLayoutEffect(()=>{
    //     resizeFuns(window.innerWidth,window.innerHeight)
    //     window.addEventListener("resize", () => {
    //         resizeFuns(window.innerWidth,window.innerHeight)
    //     });
    //     return () => {
    //         window.removeEventListener("resize", () => {})
    //     }
    // },[])

    // # Loop Frame
    useEffect(()=>{
        gl.autoClear = false;
    },[])

    // # from stable fluid - Jos Stam 1999
    // w0 -> add force -> w1 -> advect -> w2 -> diffuse -> w3 -> project -> w4 

    // # from 29a.ch
    // ┌──────────┐  vel    ┌──────────┐   vel  ┌────────────┐
    // │  advect  ├────────►│ addForce ├───────►│ divergence │
    // └────┬─────┘         └────┬─────┘        └─────┬──────┘
    //      │ vel                │  vel               │  div
    //      ▼                    ▼                    ▼
    // ┌────────────────────────────────┐        ┌────────────┐
    // │           pressure             │◄───────┤  solver    ◄─┐
    // └────────────┬───────────────────┘        └──────┬─────┘ │
    //              │ pressure                          │ pressure
    //              │                                   └───────┘
    //              ▼
    //     ┌─────────────────┐
    //     │     visualize   │
    //     └─────────────────┘
    return(
        <>

            {/* This Program Mainly for velocity caculation & error correction, it will keep in - out velocity */}
            {/* Finite difference */}
            <AdvectionSolveProgram
                scene={advectionSolveScene}
                camera={camera}
                isBounce={isBounce}
                cellScale={[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                fboSize={[screenWidth * resolution,screenHeight * resolution]}
                dt={dt}
                src={fbo_vel_0} // vel_0
                dst={fbo_vel_1} // vel_1
                isBFECC={isBFECC}
            ></AdvectionSolveProgram>

            {/* This is so called Velocity Field,in this program,it's just a small pieces of mouse area,when speed is high,it will generate more fluid */}
            {/* More color is more 'outflow' */}
            <ExternalForceProgram 
                scene={forceSolveScene}
                camera={camera}
                cellScale={[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                scale={[cursor_size,cursor_size]}
                mouse_force={mouse_force}
                dst={fbo_vel_1} // vel_1
            ></ExternalForceProgram>
            
            {/* In Experiment */}
            {/* https://jamie-wong.com/2016/08/05/webgl-fluid-simulation/ */}
            {/* 
            <FieldForceProgram 
                scene={forceSolveScene}
                camera={camera}
                dst={fbo_vel_1} // vel_1
            ></FieldForceProgram> */}

            {isViscous?<ViscousSolveProgram 
                scene={viscousSolveScene}
                camera={camera}
                iterations_viscous={iterations_viscous}
                cellScale={[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                boundarySpace={isBounce?[0,0]:[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                viscous={viscous}
                src={fbo_vel_1} // vel_1
                dst={fbo_vel_viscous_1} // vel_viscous_1
                dst_={fbo_vel_viscous_0} // vel_viscous_0
                dt={dt}
            ></ViscousSolveProgram>:<></>}

            {/* FYI , divergence causes expansion. */}
            {/* Divergence is how much fluid flows in and out.
            Divergence > 0 means more outflow and divergence <0 means more inflow. */}
            <DivergenceSolveProgram 
                scene={divergenceSolveScene}
                camera={camera}
                cellScale={[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                boundarySpace={isBounce?[0,0]:[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                src={fbo_vel_viscous_0} // ***** vel_viscous_0 *****
                dst={fbo_div} // div
                vel={isViscous?fbo_vel_viscous_1:fbo_vel_1}
                dt={dt}
            ></DivergenceSolveProgram>

            {/* Use Poisson Equation to solve pressure,because it's a iteration,so the caculation's path like a contour line */}
            <PoissonSolveProgram 
                scene={poissonSolveScene}
                camera={camera}
                cellScale={[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                boundarySpace={isBounce?[0,0]:[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                iterations_poisson={iterations_poisson}
                src={fbo_div} // div
                dst={fbo_pressure_1} // pressure_1
                dst_={fbo_pressure_0} // pressure_0
            ></PoissonSolveProgram>

            {/* Depend on pressure & velocity,to caculation in-flow & out-flow,also shaded the color */}
            <PressureSolveProgram 
                scene={pressureSolveScene}
                camera={camera}
                cellScale={[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                boundarySpace={isBounce?[0,0]:[1/(screenWidth * resolution),1/(screenHeight * resolution)]}
                src_p={fbo_pressure_0} // ****** pressure_0 ******
                src_v={fbo_vel_viscous_0} // ****** vel_viscous_0 ******
                src_update_p={fbo_pressure_1} // ****** pressure_1 ******
                src_update_v={isViscous?fbo_vel_viscous_1:fbo_vel_1}
                dst={fbo_vel_0}
                dt={dt}
            ></PressureSolveProgram>

            <ColorProgram 
                src={fbo_vel_1}
            ></ColorProgram>
        </>
    )
}


export const Effect = (props:any) =>{

    return(
      <>
          <Canvas 
            className={props.className} 
            style={{...props.style}}>
             <Perf 
                style={{position:'absolute',top:'10px',left:'10px',width:'360px',borderRadius:'10px'}}
             />
                <FluidSimulation/>
          </Canvas>
      </>
  
    )
}


