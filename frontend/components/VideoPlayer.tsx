"use client";

import { forwardRef, useImperativeHandle, useRef } from "react";

export interface VideoPlayerHandle {
  seekTo: (ts: number) => void;
}

const VideoPlayer = forwardRef<VideoPlayerHandle, { src: string }>(function VideoPlayer(
  { src },
  ref
) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useImperativeHandle(ref, () => ({
    seekTo: (ts: number) => {
      const el = videoRef.current;
      if (!el) return;
      el.currentTime = ts;
      el.play().catch(() => {
        /* autoplay can be blocked before user interaction — seeking still succeeds */
      });
    },
  }));

  return (
    <video
      ref={videoRef}
      src={src}
      controls
      className="w-full border border-hardwood-line/40 bg-hardwood"
      preload="metadata"
    />
  );
});

export default VideoPlayer;
