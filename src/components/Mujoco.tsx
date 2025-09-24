/**
 * src/components/Mujoco.tsx
 *
 * Loads MuJoCo WASM, builds the Three.js scene, steps the simulation,
 * and calls `onLoad(simulation, model)` once everything is ready.
 */

import { useFrame, useThree } from "@react-three/fiber";
import { memo, useEffect, useRef, useState } from "react";
import * as THREE from "three";

import { MujocoContainer } from "./MujocoContainer";
import {
  buildThreeScene,
  loadMujocoModule,
  loadMujocoScene,
  updateThreeScene
} from "./mujocoUtils";
import { UpdateProps } from "./UpdateProps";

// —— Import only the types we need from the WASM typings ——
// (These are "type only" imports: they do not become runtime code)
import type { Model, Simulation } from "../wasm/mujoco_wasm";

interface MujocoProps {
  sceneUrl: string;
  /**
   * Called once the MuJoCo model+state+simulation have been created
   * and the Three.js scene has been built. We only need to pass sim+model
   * to App.tsx so it can wire up the sliders.
   */
  onLoad?: (sim: Simulation, model: Model) => void;
}

export const MujocoComponent: React.FC<MujocoProps> = ({ sceneUrl, onLoad }) => {
  // If real time lags too far behind sim time, clamp it here
  const MAX_SIMULATION_LAG_MS = 35.0;

  const { scene } = useThree();

  // Prevent stepping or rendering if we're still loading
  const loadingSceneRef = useRef<boolean>(false);
  const errorRef = useRef<boolean>(false);

  // Used for stepping the simulation
  const mujocoTimeRef = useRef(0);
  const updatePropsRef = useRef<UpdateProps>();
  const tmpVecRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));

  // Robot state from API
  const robotStateRef = useRef<any>(null);
  const lastFetchTime = useRef<number>(0);
  const apiAvailable = useRef<boolean>(true);

  // Function to fetch robot state from API
  const fetchRobotState = async (): Promise<void> => {
    try {
      const response = await fetch("http://localhost:8000/api/state/full?with_head_joints=true&with_antenna_positions=true&with_body_yaw=true&with_head_pose=true");
      if (response.ok) {
        const state = await response.json();
        robotStateRef.current = state;
        apiAvailable.current = true;
      } else {
        console.warn("Failed to fetch robot state:", response.status);
        apiAvailable.current = false;
      }
    } catch (error) {
      console.warn("Error fetching robot state:", error);
      apiAvailable.current = false;
    }
  };

  // Holds the wrapper that contains WASM Module, Model, State, Simulation
  const [mujocoContainer, setMujocoContainer] = useState<MujocoContainer | null>(null);

  // 1. Load the MuJoCo WASM module once on mount
  useEffect(() => {
    const setupMujocoModule = async () => {
      try {
        const container = await loadMujocoModule();
        if (container) {
          setMujocoContainer(container);
        }
      } catch (err) {
        errorRef.current = true;
        console.error("Failed to load MuJoCo WASM:", err);
      }
    };
    setupMujocoModule();
  }, []);

  // 2. Whenever the sceneUrl or the container changes, load that MJCF into MuJoCo
  useEffect(() => {
    if (!mujocoContainer) return;

    const setupMujocoScene = async () => {
      try {
        // 2.a. Load the MJCF file (this writes it into the virtual filesystem)
        await loadMujocoScene(mujocoContainer, sceneUrl);

        // 2.b. Build the Three.js objects (meshes, materials, etc.)
        updatePropsRef.current = await buildThreeScene(mujocoContainer, scene);

        // 2.c. Once the Three.js scene is built, notify parent via onLoad
        if (onLoad) {
          const sim = mujocoContainer.getSimulation();
          const mdl = sim.model();
          onLoad(sim, mdl);
        }
      } catch (err) {
        errorRef.current = true;
        console.error("Failed to load Mujoco scene:", err);
      }
    };

    loadingSceneRef.current = true;
    setupMujocoScene()
      .finally(() => {
        loadingSceneRef.current = false;
      });
  }, [mujocoContainer, scene, sceneUrl, onLoad]);

  // 3. Every frame: step the simulation and update Three.js transforms
  useFrame(({ clock }) => {
    if (!mujocoContainer || loadingSceneRef.current || errorRef.current) {
      return;
    }

    const sim = mujocoContainer.getSimulation();
    const mdl = sim.model();
    if (!mdl || !sim) {
      return;
    }

    const timeMS = clock.getElapsedTime() * 1000;
    const timestep = mdl.getOptions().timestep;

    // ========= ROBOT API CONTROL =========
    // Adaptive polling: 1Hz when API down, 10Hz when API up
    const currentTime = clock.getElapsedTime();
    const pollInterval = apiAvailable.current ? 0.1 : 1.0; // 10Hz when up, 1Hz when down

    if (currentTime - lastFetchTime.current > pollInterval) {
      fetchRobotState();
      lastFetchTime.current = currentTime;
    }

    // Apply robot state to simulation if available
    if (robotStateRef.current && robotStateRef.current.head_joints) {
      const robotState = robotStateRef.current;

      // Map head joints to simulation controls
      for (let i = 0; i < Math.min(robotState.head_joints.length, mdl.nu - 2); i++) {
        sim.ctrl[i] = robotState.head_joints[i];
      }

      // Map antenna positions to last 2 actuators if available
      if (robotState.antennas_position && robotState.antennas_position.length >= 2 && mdl.nu >= 2) {
        const antennaStartIdx = Math.max(0, mdl.nu - 2);
        sim.ctrl[antennaStartIdx] = -robotState.antennas_position[1];     // Right antenna
        sim.ctrl[antennaStartIdx + 1] = -robotState.antennas_position[0]; // Left antenna
      }

      // Add body_yaw
      if (mdl.nu > 0) {
        sim.ctrl[0] = robotState.body_yaw;
      }
    }
    // ========= END OF ROBOT API CONTROL =========

    // Clamp if too far behind
    if (timeMS - mujocoTimeRef.current > MAX_SIMULATION_LAG_MS) {
      mujocoTimeRef.current = timeMS;
    }

    // Step until sim time catches up to real time
    while (mujocoTimeRef.current < timeMS) {
      sim.step();
      mujocoTimeRef.current += timestep * 1000;
    }

    if (!updatePropsRef.current) return;
    updateThreeScene(mujocoContainer, updatePropsRef.current, tmpVecRef.current);
  });

  // This component does not render DOM—everything is in Three.js
  return null;
};

export const Mujoco = memo(MujocoComponent);