import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, streamUrl } from "./client";

// Minimal Response-like stub so we don't depend on a real fetch implementation.
function mockResponse(opts: {
  ok: boolean;
  status: number;
  contentType?: string;
  json?: unknown;
  text?: string;
  statusText?: string;
}) {
  return {
    ok: opts.ok,
    status: opts.status,
    statusText: opts.statusText ?? "",
    headers: { get: (k: string) => (k === "content-type" ? opts.contentType ?? null : null) },
    json: async () => opts.json,
    text: async () => opts.text ?? "",
  } as unknown as Response;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("GET returns parsed JSON and prefixes /api", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(mockResponse({ ok: true, status: 200, contentType: "application/json", json: { hello: "world" } }));
    vi.stubGlobal("fetch", fetchMock);

    const data = await api.get<{ hello: string }>("/devices");
    expect(data).toEqual({ hello: "world" });
    expect(fetchMock).toHaveBeenCalledWith("/api/devices", expect.anything());
  });

  it("throws ApiError with status + body text on non-ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(mockResponse({ ok: false, status: 500, text: "boom" })),
    );

    await expect(api.get("/devices")).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
      message: "boom",
    });
    await expect(api.get("/devices")).rejects.toBeInstanceOf(ApiError);
  });

  it("returns undefined on 204 No Content", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse({ ok: true, status: 204 })));
    await expect(api.post("/cameras/x/stop")).resolves.toBeUndefined();
  });

  it("returns raw text when content-type is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse({ ok: true, status: 200, contentType: "text/plain", text: "pong" }),
      ),
    );
    await expect(api.get("/ping")).resolves.toBe("pong");
  });

  it("POST serializes a body and sets the JSON content-type", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(mockResponse({ ok: true, status: 200, contentType: "application/json", json: {} }));
    vi.stubGlobal("fetch", fetchMock);

    await api.post("/cameras/cam0/start", { fps: 30 });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({ fps: 30 }));
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("streamUrl prefixes /api without fetching", () => {
    expect(streamUrl("/cameras/cam0/stream")).toBe("/api/cameras/cam0/stream");
  });
});
