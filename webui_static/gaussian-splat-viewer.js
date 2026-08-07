/**
 * Gaussian Splat Viewer using Spark.js (THREE.js-based renderer)
 * https://github.com/sparkjsdev/spark
 *
 * Controls:
 * - W/S: Move up/down
 * - A/D: Move left/right
 * - Q/E: Decrease/increase move speed
 * - Mouse drag: Look around
 * - Scroll: Move forward/back
 * - SBS/VR: Side-by-side stereoscopic rendering
 */

class GaussianSplatViewer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.onProgress = options.onProgress || (() => {});
        this.onLoad = options.onLoad || (() => {});
        this.onError = options.onError || ((e) => console.error(e));

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        
        // Single mode
        this.splat = null;
        
        // Video mode
        this.frameMeshes = [];
        this.currentFrameIndex = -1;

        this.animationId = null;

        // First-person controls state
        this.keys = {};
        this.mouseDown = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.yaw = Math.PI;    // Start facing toward -Z (toward the splat)
        this.pitch = 0;
        this.moveSpeed = 2.0;  // Units per second
        this.speedScaleStep = 1.2;
        this.wheelMoveStep = 0.35;
        this.lookSpeed = 0.003;
        this.lastTime = performance.now();
        this.interactionAxes = null;
        this.interactionAxesHideAt = 0;
        this.interactionAxesDistance = 1.2;

        // VR / Stereo State
        this.stereoMode = false;
        this.stereoCamera = null;

        // Create a promise that resolves when init is complete
        this.ready = this.init();
        // Some callers attach after creating the canvas; prevent a transient unhandled rejection.
        this.ready.catch(() => {});
    }

    async init() {
        try {
            // Import THREE.js and Spark dynamically
            const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js');
            const { SplatMesh } = await import('https://sparkjs.dev/releases/spark/0.1.10/spark.module.js');

            this.THREE = THREE;
            this.SplatMesh = SplatMesh;

            // Setup renderer
            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvas,
                antialias: true,
                alpha: false
            });
            this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.setClearColor(0x000000, 1);
            
            // Required for partial viewport rendering in VR mode
            this.renderer.setScissorTest(false);

            // Setup scene
            this.scene = new THREE.Scene();

            // Setup camera
            this.camera = new THREE.PerspectiveCamera(
                60,
                this.canvas.clientWidth / this.canvas.clientHeight,
                0.01,
                1000
            );
            this.camera.position.set(0, 0, 3);

            this.createInteractionAxes();

            // Setup Stereo Camera for VR
            this.stereoCamera = new THREE.StereoCamera();
            this.stereoCamera.aspect = 0.5;

            // Setup first-person controls
            this.setupControls();

            // Handle resize
            this.resizeObserver = new ResizeObserver(() => this.handleResize());
            this.resizeObserver.observe(this.canvas);

            // Start render loop
            this.animate();
        } catch (error) {
            console.error('GaussianSplatViewer: Failed to initialize:', error);
            throw error;
        }
    }

    setupControls() {
        this.keyDownHandler = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
            const key = e.key.toLowerCase();
            if (key === 'q') {
                this.adjustMoveSpeed(1 / this.speedScaleStep);
                e.preventDefault();
                return;
            }
            if (key === 'e') {
                this.adjustMoveSpeed(this.speedScaleStep);
                e.preventDefault();
                return;
            }
            this.keys[key] = true;
        };
        this.keyUpHandler = (e) => {
            this.keys[e.key.toLowerCase()] = false;
        };

        this.mouseDownHandler = (e) => {
            if (e.target !== this.canvas) return;
            e.preventDefault();
            this.mouseDown = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
            this.showInteractionAxes(1200);
        };
        this.mouseUpHandler = () => {
            if (!this.mouseDown) return;
            this.mouseDown = false;
            this.canvas.style.cursor = 'grab';
            this.showInteractionAxes(700);
        };
        this.mouseMoveHandler = (e) => {
            if (!this.mouseDown) return;
            e.preventDefault();
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            this.yaw += deltaX * this.lookSpeed;
            this.pitch -= deltaY * this.lookSpeed;
            this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            if (deltaX !== 0 || deltaY !== 0) {
                this.showInteractionAxes(900);
            }
        };

        this.wheelHandler = (e) => {
            if (e.target !== this.canvas) return;
            e.preventDefault();
            if (!this.camera) return;
            const direction = new this.THREE.Vector3();
            this.camera.getWorldDirection(direction);
            const wheelDirection = e.deltaY < 0 ? 1 : -1;
            this.camera.position.add(direction.multiplyScalar(this.wheelMoveStep * wheelDirection));
            this.showInteractionAxes(900);
        };

        document.addEventListener('keydown', this.keyDownHandler);
        document.addEventListener('keyup', this.keyUpHandler);
        this.canvas.addEventListener('mousedown', this.mouseDownHandler);
        document.addEventListener('mouseup', this.mouseUpHandler);
        document.addEventListener('mousemove', this.mouseMoveHandler);
        this.canvas.addEventListener('wheel', this.wheelHandler, { passive: false });
        this.canvas.style.cursor = 'grab';
    }

    adjustMoveSpeed(multiplier) {
        this.moveSpeed *= multiplier;
        this.moveSpeed = Math.max(0.1, Math.min(20, this.moveSpeed));
        this.showInteractionAxes(900);
    }

    createInteractionAxes() {
        if (!this.THREE || !this.scene) return;

        const THREE = this.THREE;
        const group = new THREE.Group();
        group.visible = false;
        group.renderOrder = 1000;

        const axisLength = 0.07;
        const axisRadius = 0.0045;
        const axes = [
            { direction: new THREE.Vector3(1, 0, 0), color: 0xff5a5a },
            { direction: new THREE.Vector3(0, 1, 0), color: 0x52d273 },
            { direction: new THREE.Vector3(0, 0, 1), color: 0x60a5fa },
        ];

        axes.forEach(({ direction, color }) => {
            const geometry = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 10);
            const material = new THREE.MeshBasicMaterial({
                color,
                transparent: true,
                opacity: 0.95,
                depthTest: false,
                depthWrite: false,
            });
            const axis = new THREE.Mesh(geometry, material);
            axis.position.copy(direction.clone().multiplyScalar(axisLength / 2));
            axis.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
            axis.renderOrder = 1000;
            group.add(axis);
        });

        this.scene.add(group);
        this.interactionAxes = group;
    }

    showInteractionAxes(durationMs = 900) {
        if (!this.interactionAxes || !this.camera) return;
        this.interactionAxes.visible = true;
        this.interactionAxesHideAt = performance.now() + durationMs;
        this.updateInteractionAxes();
    }

    hideInteractionAxes() {
        if (!this.interactionAxes) return;
        this.interactionAxes.visible = false;
        this.interactionAxesHideAt = 0;
    }

    hasActiveMovementKeys() {
        return Boolean(this.keys['w'] || this.keys['s'] || this.keys['a'] || this.keys['d']);
    }

    updateInteractionAxes() {
        if (!this.interactionAxes || !this.camera) return;

        const forward = new this.THREE.Vector3();
        this.camera.getWorldDirection(forward);
        this.interactionAxes.position.copy(this.camera.position).add(forward.multiplyScalar(this.interactionAxesDistance));
        this.interactionAxes.quaternion.identity();
    }

    handleResize() {
        if (!this.renderer || !this.camera) return;
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);

        // Ensure we reset viewport if we exit stereo mode during resize
        if (!this.stereoMode) {
            this.renderer.setScissor(0, 0, width, height);
            this.renderer.setViewport(0, 0, width, height);
        }
    }

    updateControls(deltaTime) {
        if (!this.camera) return;
        const right = new this.THREE.Vector3(Math.cos(this.yaw), 0, Math.sin(this.yaw));
        const up = new this.THREE.Vector3(0, -1, 0);
        const velocity = new this.THREE.Vector3();
        const speed = this.moveSpeed * deltaTime;

        if (this.keys['w']) velocity.add(up.clone().multiplyScalar(speed));
        if (this.keys['s']) velocity.add(up.clone().multiplyScalar(-speed));
        if (this.keys['a']) velocity.add(right.clone().multiplyScalar(speed));
        if (this.keys['d']) velocity.add(right.clone().multiplyScalar(-speed));

        this.camera.position.add(velocity);
        if (velocity.lengthSq() > 0) {
            this.showInteractionAxes(250);
        }
        const quaternion = new this.THREE.Quaternion();
        const euler = new this.THREE.Euler(-this.pitch, this.yaw, Math.PI, 'YXZ');
        quaternion.setFromEuler(euler);
        this.camera.quaternion.copy(quaternion);
    }

    toggleStereo() {
        if (!this.renderer) return;
        this.stereoMode = !this.stereoMode;
        
        if (this.stereoMode) {
            this.renderer.setScissorTest(true);
            this.handleResize();
        } else {
            this.renderer.setScissorTest(false);
            this.handleResize(); // Reset full viewport
        }
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        const now = performance.now();
        // Avoid a large movement jump after the window has been suspended.
        const deltaTime = Math.min((now - this.lastTime) / 1000, 0.1);
        this.lastTime = now;
        
        this.updateControls(deltaTime);

        if (this.interactionAxes && this.interactionAxes.visible) {
            this.updateInteractionAxes();
            if (now > this.interactionAxesHideAt && !this.mouseDown && !this.hasActiveMovementKeys()) {
                this.hideInteractionAxes();
            }
        }
        
        if (this.renderer && this.scene && this.camera) {
            if (this.stereoMode && this.stereoCamera) {
                // VR / SBS Mode
                // Three.js viewport/scissor APIs take CSS pixels, not the
                // device-pixel-ratio-scaled drawing-buffer size. canvas.width
                // would push the right eye to ~75% on 1.5x DPI displays.
                const width = this.canvas.clientWidth;
                const height = this.canvas.clientHeight;
                const halfWidth = Math.floor(width / 2);

                // StereoCamera is rendered directly, so keep the main camera matrix current.
                this.camera.updateMatrixWorld();

                // Sync the stereo camera rig with the main camera position/rotation
                this.stereoCamera.update(this.camera);

                // Render Left Eye
                this.renderer.setScissor(0, 0, halfWidth, height);
                this.renderer.setViewport(0, 0, halfWidth, height);
                this.renderer.render(this.scene, this.stereoCamera.cameraL);

                // Render Right Eye
                this.renderer.setScissor(halfWidth, 0, width - halfWidth, height);
                this.renderer.setViewport(halfWidth, 0, width - halfWidth, height);
                this.renderer.render(this.scene, this.stereoCamera.cameraR);
            } else {
                // Standard Single Mode
                this.renderer.render(this.scene, this.camera);
            }
        }
    }

    /**
     * Preloads frames and keeps only the active frame attached to the scene.
     */
    async preloadFrames(urls, progressCallback) {
        await this.ready;
        this.clearScene();
        this.clearFrameMeshes();

        try {
            for (let i = 0; i < urls.length; i++) {
                const url = urls[i];

                const mesh = new this.SplatMesh({
                    url: url,
                    visible: false
                });

                // Wait for parsing
                await mesh.loadPromise;

                mesh.visible = false;
                this.frameMeshes.push(mesh);

                // Free the source blob once Spark has parsed it -- holding every
                // ~64MB frame blob at once is what exhausts memory. No-op for the
                // server-URL path (not a blob:).
                if (typeof url === 'string' && url.startsWith('blob:')) {
                    URL.revokeObjectURL(url);
                }

                // Center camera on first frame only
                if (i === 0) {
                    this.centerCameraOnSplat();
                }

                if (progressCallback) {
                    progressCallback(i + 1, urls.length);
                }
            }

            this.showFrame(0);
            this.onLoad();

        } catch (error) {
            console.error("Error preloading frames:", error);
            this.clearFrameMeshes();
            throw error;
        }
    }

    async warmUpFrames(progressCallback) {
        await this.ready;
        if (!this.frameMeshes || this.frameMeshes.length === 0) return;

        const restoreIndex = this.currentFrameIndex >= 0 ? this.currentFrameIndex : 0;

        for (let i = 0; i < this.frameMeshes.length; i++) {
            this.showFrame(i);
            await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            if (progressCallback) {
                progressCallback(i + 1, this.frameMeshes.length);
            }
        }

        this.showFrame(restoreIndex);
        await new Promise((resolve) => requestAnimationFrame(resolve));
    }

    /**
     * Switch to the given frame (all frames are already parsed and in VRAM).
     */
    showFrame(index) {
        if (!this.frameMeshes || this.frameMeshes.length === 0) return;
        if (index < 0 || index >= this.frameMeshes.length) return;

        const nextMesh = this.frameMeshes[index];
        if (!nextMesh) return;

        if (this.currentFrameIndex === index && nextMesh.parent === this.scene) {
            return;
        }

        const currentMesh = this.frameMeshes[this.currentFrameIndex];
        if (currentMesh) {
            currentMesh.visible = false;
            if (currentMesh.parent === this.scene) {
                this.scene.remove(currentMesh);
            }
        }

        nextMesh.visible = true;
        if (nextMesh.parent !== this.scene) {
            this.scene.add(nextMesh);
        }
        this.currentFrameIndex = index;
    }

    /**
     * Legacy single file loader
     */
    async loadPly(url, centerCamera = true) {
        try {
            await this.ready;

            // Clear video data if any
            this.clearFrameMeshes();

            if (centerCamera) this.onProgress(10);

            const newSplat = new this.SplatMesh({
                url: url,
                onProgress: (progress) => {
                    if (centerCamera) this.onProgress(20 + progress * 70);
                }
            });

            await newSplat.loadPromise;

            if (centerCamera) this.onProgress(95);

            this.scene.add(newSplat);

            if (this.splat) {
                const old = this.splat;
                setTimeout(() => {
                    this.scene.remove(old);
                    if(old.dispose) old.dispose();
                }, 100);
            }
            this.splat = newSplat;

            if (centerCamera) {
                this.centerCameraOnSplat();
            }

            if (centerCamera) this.onProgress(100);
            this.onLoad();

        } catch (error) {
            console.error('GaussianSplatViewer: Failed to load PLY:', error);
            this.onError(error);
        }
    }

    clearScene() {
        if (this.splat) {
            this.scene.remove(this.splat);
            if (this.splat.dispose) this.splat.dispose();
            this.splat = null;
        }
    }

    clearFrameMeshes() {
        if (this.frameMeshes.length === 0) {
            this.currentFrameIndex = -1;
            return;
        }

        this.frameMeshes.forEach(mesh => {
            if (mesh.parent === this.scene) {
                this.scene.remove(mesh);
            }
            if (mesh.dispose) mesh.dispose();
        });
        this.frameMeshes = [];
        this.currentFrameIndex = -1;
    }

    centerCameraOnSplat() {
        if (!this.camera) return;
        const distance = 3.0;
        this.camera.position.set(0, 0, distance);
        this.yaw = Math.PI;
        this.pitch = 0;
    }

    resetCamera() {
        this.centerCameraOnSplat();
        if (this.camera) this.camera.position.z -= 3.0;
        this.showInteractionAxes(900);
    }

    disposeObject3D(object) {
        if (!object) return;
        object.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach((material) => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        document.removeEventListener('keydown', this.keyDownHandler);
        document.removeEventListener('keyup', this.keyUpHandler);
        this.canvas.removeEventListener('mousedown', this.mouseDownHandler);
        document.removeEventListener('mouseup', this.mouseUpHandler);
        document.removeEventListener('mousemove', this.mouseMoveHandler);
        this.canvas.removeEventListener('wheel', this.wheelHandler);

        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        
        this.clearScene();
        this.clearFrameMeshes();

        if (this.interactionAxes) {
            if (this.scene) this.scene.remove(this.interactionAxes);
            this.disposeObject3D(this.interactionAxes);
            this.interactionAxes = null;
        }

        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }
        this.scene = null;
        this.camera = null;
        this.stereoCamera = null;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GaussianSplatViewer;
}
