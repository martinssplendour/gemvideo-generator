from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, send_from_directory

from config import load_settings
from routes import (
    create_v2_jobs_blueprint,
    create_v2_projects_blueprint,
    create_v2_voices_blueprint,
)
from services import (
    SimpleVideoService,
    StorageService,
    VeoClientService,
    VeoDirectorService,
    VideoFrameService,
)
from services.local_render_runner import LocalRenderRunner
from services.v2_runtime import build_v2_services


def create_app() -> Flask:
    load_dotenv()
    settings = load_settings()

    storage_service = StorageService(
        output_dir=settings.output_dir,
        upload_dir=settings.upload_dir,
    )
    director_service = None
    if settings.api_key:
        frame_service = VideoFrameService(output_dir=settings.output_dir)
        veo_client_service = VeoClientService(
            api_key=settings.api_key,
            model_name=settings.model_name,
            poll_interval_seconds=settings.poll_interval_seconds,
        )
        director_service = VeoDirectorService(
            storage_service=storage_service,
            frame_service=frame_service,
            veo_client_service=veo_client_service,
            max_retries=settings.max_retries,
            retry_delay_seconds=settings.retry_delay_seconds,
        )
    v2_services = build_v2_services(settings)
    local_render_runner = LocalRenderRunner(
        manifest_service=v2_services.manifest_service,
        render_service=v2_services.render_service,
    )
    simple_video_service = SimpleVideoService(
        storage_service=storage_service,
        video_client_service=v2_services.render_service.veo_client_service,
        provider=settings.video_provider,
        model_name=(
            settings.openai_video_model
            if settings.video_provider == "openai"
            else settings.model_name
        ),
    )

    app = Flask(
        __name__,
        static_folder=settings.static_folder,
        static_url_path="",
    )
    app.config["OUTPUT_DIR"] = settings.output_dir
    app.config["DIRECTOR_SERVICE"] = director_service
    app.config["STORAGE_SERVICE"] = storage_service
    app.config["V2_SERVICES"] = v2_services
    app.config["SIMPLE_VIDEO_SERVICE"] = simple_video_service

    app.register_blueprint(
        create_v2_projects_blueprint(
            manifest_service=v2_services.manifest_service,
            validation_service=v2_services.validation_service,
            render_runner=local_render_runner,
        )
    )
    app.register_blueprint(
        create_v2_jobs_blueprint(
            manifest_service=v2_services.manifest_service,
            settings=settings,
        )
    )
    app.register_blueprint(
        create_v2_voices_blueprint(
            manifest_service=v2_services.manifest_service,
            voice_service=v2_services.voice_service,
        )
    )

    @app.route("/")
    def index():
        return send_file("index.html")

    @app.route("/output/<path:filename>")
    def serve_output(filename: str):
        return send_from_directory(app.config["OUTPUT_DIR"], filename)

    @app.route("/api/generate", methods=["POST"])
    def handle_generate():
        try:
            prompt = (request.form.get("prompt") or "").strip()
            if not prompt:
                return jsonify({"error": "No prompt provided"}), 400

            storage: StorageService = app.config["STORAGE_SERVICE"]
            director: VeoDirectorService = app.config["DIRECTOR_SERVICE"]
            if director is None:
                return jsonify(
                    {
                        "error": (
                            "Legacy /api/generate requires GEMINI_API_KEY because it uses Veo directly. "
                            "Use /api/v2/projects render flow for VIDEO_PROVIDER=openai."
                        )
                    }
                ), 400

            uploaded_files = request.files.getlist("ref_images")
            uploaded_paths = storage.save_uploads(uploaded_files)
            if uploaded_paths:
                director.add_references(uploaded_paths)

            result = director.generate_clip(prompt)
            return jsonify(
                {
                    "videoUrl": f"/output/{result.clip_filename}",
                    "endFrameUrl": f"/output/{result.end_frame_filename}",
                    "clipId": result.clip_id,
                    "activeRefCount": result.active_reference_count,
                }
            )
        except Exception as error:
            print(f"Final Error: {error}")
            return jsonify({"error": str(error)}), 500

    @app.route("/api/simple-video", methods=["POST"])
    def handle_simple_video():
        payload = request.get_json(silent=True) or {}
        prompt = (request.form.get("prompt") or payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        image = (
            request.files.get("image")
            or request.files.get("reference_image")
            or request.files.get("ref_image")
        )
        simple_video: SimpleVideoService = app.config["SIMPLE_VIDEO_SERVICE"]
        try:
            result = simple_video.generate_video(prompt=prompt, image_upload=image)
            return jsonify(
                {
                    "videoUrl": f"/output/{result.output_filename}",
                    "filename": result.output_filename,
                    "provider": result.provider,
                    "model": result.model_name,
                    "hasReferenceImage": result.used_reference_image,
                    "aspectRatio": result.aspect_ratio,
                }
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception as error:
            print(f"Simple video generation error: {error}")
            return jsonify({"error": str(error)}), 500

    @app.errorhandler(404)
    def handle_404(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Endpoint not found.", "path": request.path}), 404
        return error

    @app.errorhandler(413)
    def handle_413(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Uploaded file is too large."}), 413
        return error

    @app.errorhandler(405)
    def handle_405(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Method not allowed for this API endpoint."}), 405
        return error

    @app.errorhandler(500)
    def handle_500(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error."}), 500
        return error

    return app


app = create_app()

if __name__ == "__main__":
    print("Starting Veo3 Director Server...")
    app.run(debug=True, port=5000, use_reloader=False)
