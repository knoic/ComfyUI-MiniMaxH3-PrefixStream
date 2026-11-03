"""Filesystem boundary and HTTP regression tests for timeline video access."""

import asyncio
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from engine.clip_bin_api import register_clip_bin_routes
from engine.timeline_session_manager import resolve_video_path
from engine.video_path_security import validate_video_path


class TestVideoPathSecurity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.output = self.root / "output"
        self.input.mkdir()
        self.output.mkdir()
        self.video = self.input / "clip.MP4"
        self.video.write_bytes(b"video fixture")
        self.outside = self.root / "secret.mp4"
        self.outside.write_bytes(b"outside fixture")
        self.secret = self.input / ".env"
        self.secret.write_bytes(b"secret fixture")
        folder_paths = types.SimpleNamespace(
            get_input_directory=lambda: str(self.input),
            get_output_directory=lambda: str(self.output),
        )
        self.modules = patch.dict(sys.modules, {"folder_paths": folder_paths})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_allowed_input_output_and_clip_bins(self):
        for path in (self.video, self.output / "export.mov", self.output / "minimax_h3_bins" / "project" / "clip.webm"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self.assertEqual(resolve_video_path(str(path)), os.path.realpath(path))
        self.assertEqual(resolve_video_path("clip.MP4"), os.path.realpath(self.video))

    def test_reject_absolute_traversal_sibling_and_nonvideo(self):
        sibling = self.root / "input-other" / "clip.mp4"
        sibling.parent.mkdir()
        sibling.touch()
        for path in (str(self.outside), "../secret.mp4", str(sibling), str(self.secret)):
            self.assertEqual(resolve_video_path(path), "")
            with self.assertRaises(ValueError):
                validate_video_path(str(self.input / path))

    def test_different_drive_does_not_skip_later_root(self):
        target = self.output / "clip.mp4"
        target.touch()
        commonpath = os.path.commonpath
        with patch("engine.video_path_security.os.path.commonpath", side_effect=[ValueError("different drives"), os.path.normcase(commonpath([str(self.output), str(target)]))]):
            self.assertEqual(validate_video_path(str(target)), os.path.realpath(target))

    def test_symlink_escape(self):
        link = self.input / "linked.mp4"
        try:
            link.symlink_to(self.outside)
        except OSError as exc:
            self.skipTest(f"Symlinks unavailable: {exc}")
        self.assertEqual(resolve_video_path(str(link)), "")
        with self.assertRaises(ValueError):
            validate_video_path(str(link))

    def test_canonical_link_target_is_checked(self):
        link = self.input / "linked.mp4"
        realpath = os.path.realpath
        def resolve(path):
            return realpath(self.outside if str(path) == str(link) else path)
        with patch("engine.video_path_security.os.path.realpath", side_effect=resolve):
            with self.assertRaises(ValueError):
                validate_video_path(str(link))

    def test_local_node_opt_in_and_missing_configuration(self):
        self.assertEqual(resolve_video_path(str(self.outside), allow_external=True), os.path.realpath(self.outside))
        with patch.dict(sys.modules, {"folder_paths": None}):
            self.assertEqual(resolve_video_path(str(self.video)), "")

    def test_http_stream_and_probe_boundary(self):
        async def check():
            routes = web.RouteTableDef()
            server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes)))
            with patch.dict(sys.modules, {"server": server}):
                register_clip_bin_routes()
            app = web.Application()
            app.add_routes(routes)
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/minimax/timeline/video_stream", params={"file": str(self.video)})
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.read(), b"video fixture")
                response = await client.get("/minimax/timeline/video_stream", params={"file": str(self.input / "missing.mp4")})
                self.assertEqual(response.status, 404)
                for path in (str(self.outside), str(self.secret), str(self.input / ".." / "secret.mp4")):
                    response = await client.get("/minimax/timeline/video_stream", params={"file": path})
                    self.assertEqual(response.status, 403)
                with patch("engine.clip_bin_api.get_or_create_timeline_session") as create_session:
                    for path in (str(self.outside), "../secret.mp4", str(self.secret)):
                        response = await client.get("/minimax/timeline/probe_video", params={"video": path})
                        self.assertEqual(response.status, 404)
                    create_session.assert_not_called()
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
