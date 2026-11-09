#!/usr/bin/env python3
"""Localhost API for real SmolVLA inference and a native LIBERO session.

All GPU and EGL work runs on one dedicated thread, including initialization.
No dummy action/model fallback is used when weights or native assets are missing.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import os
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, ConfigDict, Field

try:
    from .runtime import NativeRuntime, NotReady, SessionConflict
except ImportError:
    from runtime import NativeRuntime, NotReady, SessionConflict


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ResetRequest(Request):
    initial_state_id: int = Field(default=0, ge=0, le=49)
    seed: int = Field(default=123, ge=0)
    control_source: Literal['policy', 'operator'] = 'policy'
    layout_seed: int | None = Field(default=None, ge=0)


class SessionRequest(Request):
    session_id: str = Field(min_length=1, max_length=80)


class ControlRequest(SessionRequest):
    source: Literal['policy', 'operator']


class StepRequest(ControlRequest):
    action: list[float] | None = None
    steps: int = Field(default=1, ge=1, le=10)


class MarkRequest(SessionRequest):
    marker: Literal['recovery_complete', 'note']
    detail: str = Field(default='', max_length=2000)


class PredictRequest(Request):
    camera1: str = Field(min_length=16, max_length=4_000_000)
    camera2: str = Field(min_length=16, max_length=4_000_000)
    state: list[float] = Field(min_length=8, max_length=8)
    task: str = Field(min_length=1, max_length=2000)
    reset: bool = True


def create_app(runtime_factory=NativeRuntime) -> FastAPI:
    runtime = runtime_factory()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-libero')

    async def invoke(method, *args):
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(executor, method, *args)
        except NotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SessionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            runtime.last_api_error = f'{type(exc).__name__}: {exc}'
            raise HTTPException(status_code=503, detail=runtime.last_api_error) from exc

    @asynccontextmanager
    async def lifespan(app):
        loading = None
        if os.environ.get('WERESCUE_AUTOLOAD', '1') != '0':
            loading = asyncio.create_task(invoke(runtime.load))
            # Errors remain visible in /health; do not leave an unobserved task exception.
            loading.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        yield
        if loading and not loading.done():
            await loading
        await invoke(runtime.close)
        executor.shutdown(wait=True)

    app = FastAPI(title='WeRescue native policy bridge', version='1.0.0', lifespan=lifespan)
    # Exact native mesh/texture bundles are large. Only compress large replies;
    # ordinary JPEG telemetry frames avoid an extra compression pass.
    app.add_middleware(GZipMiddleware, minimum_size=1_000_000, compresslevel=3)
    origins = os.environ.get('WERESCUE_CORS_ORIGINS',
        'http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173').split(',')
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'Authorization'])

    async def authorize(authorization: str | None = Header(default=None)):
        token = os.environ.get('WERESCUE_API_TOKEN', '')
        if token and not hmac.compare_digest(authorization or '', 'Bearer ' + token):
            raise HTTPException(status_code=401, detail='Bearer token required')

    @app.get('/health', dependencies=[Depends(authorize)])
    async def health():
        return runtime.health()

    @app.post('/session/reset', dependencies=[Depends(authorize)])
    async def reset(req: ResetRequest):
        return await invoke(runtime.reset_session, req.initial_state_id, req.seed, req.control_source, req.layout_seed)

    @app.get('/session/frame', dependencies=[Depends(authorize)])
    async def frame(session_id: str):
        return await invoke(runtime.get_frame, session_id)

    @app.get('/scene', dependencies=[Depends(authorize)])
    @app.get('/session/scene', dependencies=[Depends(authorize)])
    async def scene(session_id: str):
        return await invoke(runtime.scene, session_id)

    @app.post('/session/control', dependencies=[Depends(authorize)])
    async def control(req: ControlRequest):
        return await invoke(runtime.control, req.session_id, req.source)

    @app.post('/session/step', dependencies=[Depends(authorize)])
    async def step(req: StepRequest):
        return await invoke(runtime.step, req.session_id, req.source, req.action, req.steps)

    @app.post('/session/mark', dependencies=[Depends(authorize)])
    async def mark(req: MarkRequest):
        return await invoke(runtime.mark, req.session_id, req.marker, req.detail)

    @app.get('/session/export', dependencies=[Depends(authorize)])
    async def export(session_id: str):
        return await invoke(runtime.export, session_id)

    @app.post('/predict', dependencies=[Depends(authorize)])
    async def predict(req: PredictRequest):
        return await invoke(runtime.predict, req.camera1, req.camera2, req.state, req.task, req.reset)

    return app


if os.environ.get('WERESCUE_MODEL_ARM') == 'base-adapted':
    try:
        from .base_runtime import BaseAdaptedRuntime
    except ImportError:
        from base_runtime import BaseAdaptedRuntime
    app = create_app(runtime_factory=BaseAdaptedRuntime)
else:
    app = create_app()

if __name__ == '__main__':
    import uvicorn
    host = os.environ.get('WERESCUE_HOST', '127.0.0.1')
    if host not in ('127.0.0.1', 'localhost', '::1') and not os.environ.get('WERESCUE_API_TOKEN'):
        raise RuntimeError('A non-loopback bind requires WERESCUE_API_TOKEN')
    uvicorn.run(app, host=host,
                port=int(os.environ.get('WERESCUE_PORT', '8000')))
