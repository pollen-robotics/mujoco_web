import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { DepthOfField, EffectComposer } from "@react-three/postprocessing";
import * as THREE from "three";
import "./App.css";
import { Mujoco } from "./components/Mujoco";
import "./index.css";

const App = () => {
    return (
        <Canvas
            tabIndex={-1}                                  // disable focus outline
            shadows="soft"
            dpr={window.devicePixelRatio}
            onCreated={(state) => {
                state.scene.background = new THREE.Color(0x264059);
            }}
            style={{
                position: "fixed",                           // cover the viewport
                top: 0,
                left: 0,
                width: "100vw",
                height: "100vh",
                outline: "none",                              // extra safety against focus rings
            }}
        >
            {/* lights + camera/controllers */}
            <ambientLight color={0xffffff} intensity={0.1} />
            <spotLight
                position={[0, 2, 2]}
                angle={0.15}
                penumbra={1}
                decay={1}
                intensity={3.14}
            />
            <PerspectiveCamera makeDefault position={[2.0, 1.7, 1.7]} fov={45} />
            <OrbitControls makeDefault />

            {/* MuJoCo scene */}
            <Mujoco sceneUrl={"reachy/scenes/empty.xml"} />

            {/* post‐processing */}
            <EffectComposer>
                <DepthOfField
                    focusDistance={0}
                    focalLength={0.02}
                    bokehScale={2}
                    height={480}
                />
            </EffectComposer>
        </Canvas>
    );
};

export default App;
