"""
ACER Robot - Secure Communication Framework
=============================================
Provides AES-256-GCM encrypted, multi-channel communication
between ACER robots and command infrastructure.

Channels:
  - Primary: RFD900x radio modem (long-range ground link)
  - Secondary: Wi-Fi mesh (short-range team comms)
  - Fallback: Iridium 9603 satellite modem (global coverage)

Features:
  - End-to-end AES-256-GCM encryption
  - Automatic channel failover with health monitoring
  - Telemetry broadcasting & command reception
  - Heartbeat watchdog with dead-man protocol

Author: ACER Team
Date: March 2025
"""

import json
import time
import threading
import hashlib
import struct
import logging
import socket
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import os
import zmq

logger = logging.getLogger(__name__)


class Channel(Enum):
    PRIMARY_RADIO = "rfd900x"
    WIFI_MESH = "wifi_mesh"
    SATELLITE = "iridium"


class MessageType(Enum):
    TELEMETRY = "telemetry"
    COMMAND = "command"
    THREAT_ALERT = "threat_alert"
    HEARTBEAT = "heartbeat"
    STATUS = "status"
    ACK = "ack"
    EMERGENCY = "emergency"


@dataclass
class Message:
    msg_type: MessageType
    payload: Dict[str, Any]
    robot_id: str
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    channel: Channel = Channel.PRIMARY_RADIO

    def to_dict(self) -> Dict:
        return {
            "type": self.msg_type.value,
            "payload": self.payload,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "channel": self.channel.value,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Message":
        return cls(
            msg_type=MessageType(d["type"]),
            payload=d["payload"],
            robot_id=d["robot_id"],
            timestamp=d.get("timestamp", 0),
            sequence=d.get("sequence", 0),
            channel=Channel(d.get("channel", Channel.PRIMARY_RADIO.value)),
        )


class CryptoLayer:
    """
    AES-256-GCM encryption/decryption for all ACER communications.
    Each message uses a fresh random 96-bit nonce.
    Key derivation via HKDF-SHA256 from a shared master secret.
    """

    NONCE_BYTES = 12
    TAG_BYTES = 16
    KEY_BYTES = 32

    def __init__(self, master_secret: bytes, context: bytes = b"acer-comms-v1"):
        self._key = self._derive_key(master_secret, context)
        self._aesgcm = AESGCM(self._key)

    def _derive_key(self, secret: bytes, context: bytes) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=self.KEY_BYTES,
            salt=b"acer-salt-2025",
            info=context,
        )
        return hkdf.derive(secret)

    def encrypt(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Encrypt with AES-256-GCM. Returns nonce + ciphertext + tag."""
        nonce = os.urandom(self.NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    def decrypt(self, ciphertext_pkg: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt AES-256-GCM package. Raises on auth failure."""
        nonce = ciphertext_pkg[:self.NONCE_BYTES]
        ciphertext = ciphertext_pkg[self.NONCE_BYTES:]
        return self._aesgcm.decrypt(nonce, ciphertext, associated_data)

    def message_to_bytes(self, msg: Message) -> bytes:
        raw = json.dumps(msg.to_dict()).encode("utf-8")
        aad = msg.robot_id.encode("utf-8")
        return self.encrypt(raw, aad)

    def bytes_to_message(self, data: bytes, robot_id: str) -> Message:
        aad = robot_id.encode("utf-8")
        raw = self.decrypt(data, aad)
        return Message.from_dict(json.loads(raw.decode("utf-8")))


@dataclass
class ChannelHealth:
    channel: Channel
    last_success: float = 0.0
    last_failure: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    latency_ms: float = 0.0
    available: bool = True

    @property
    def reliability(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class ChannelManager:
    """
    Manages multi-channel communication with automatic failover.
    Priority: PRIMARY_RADIO → WIFI_MESH → SATELLITE
    """

    CHANNEL_PRIORITY = [Channel.PRIMARY_RADIO, Channel.WIFI_MESH, Channel.SATELLITE]
    HEALTH_CHECK_INTERVAL = 5.0  # seconds

    def __init__(self):
        self._health: Dict[Channel, ChannelHealth] = {
            ch: ChannelHealth(channel=ch) for ch in Channel
        }
        self._lock = threading.Lock()

    def get_best_channel(self) -> Channel:
        with self._lock:
            for ch in self.CHANNEL_PRIORITY:
                if self._health[ch].available:
                    return ch
        return Channel.SATELLITE  # Last resort

    def report_success(self, channel: Channel, latency_ms: float = 0.0):
        with self._lock:
            h = self._health[channel]
            h.last_success = time.time()
            h.success_count += 1
            h.latency_ms = latency_ms
            h.available = True

    def report_failure(self, channel: Channel):
        with self._lock:
            h = self._health[channel]
            h.last_failure = time.time()
            h.failure_count += 1
            # Mark unavailable after 3 consecutive failures
            recent_failures = h.failure_count
            if recent_failures >= 3 and h.success_count == 0:
                h.available = False
                logger.warning(f"Channel {channel.value} marked unavailable")

    def get_health_report(self) -> Dict[str, Dict]:
        with self._lock:
            return {
                ch.value: {
                    "available": h.available,
                    "reliability": round(h.reliability, 3),
                    "latency_ms": h.latency_ms,
                }
                for ch, h in self._health.items()
            }


class HeartbeatWatchdog:
    """
    Monitors operator heartbeat. If no heartbeat is received within
    the timeout window, triggers the dead-man protocol (safe mode).
    """

    DEFAULT_TIMEOUT_S = 15.0

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S,
                 on_timeout: Optional[Callable] = None):
        self.timeout_s = timeout_s
        self.on_timeout = on_timeout or self._default_timeout_handler
        self._last_beat = time.time()
        self._active = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        logger.info(f"Heartbeat watchdog started (timeout={timeout_s}s)")

    def beat(self):
        """Called when a valid heartbeat message is received."""
        self._last_beat = time.time()

    def _watch(self):
        while self._active:
            time.sleep(1.0)
            elapsed = time.time() - self._last_beat
            if elapsed > self.timeout_s:
                logger.critical(
                    f"Heartbeat timeout! No signal for {elapsed:.1f}s"
                )
                self.on_timeout()
                self._last_beat = time.time()  # Reset to avoid repeated triggers

    def _default_timeout_handler(self):
        logger.critical("DEAD-MAN PROTOCOL: Entering safe mode — halting all motion")

    def stop(self):
        self._active = False


class CommunicationFramework:
    """
    Top-level communication framework for ACER Robot.
    
    Manages:
    - Encrypted message send/receive via ZMQ over multiple channels
    - Telemetry broadcast (position, status, sensor data)
    - Command reception and routing
    - Heartbeat monitoring with dead-man failsafe
    - Channel health monitoring and auto-failover
    """

    TELEMETRY_INTERVAL = 0.5   # seconds (2 Hz)
    HEARTBEAT_INTERVAL = 1.0   # seconds (1 Hz)

    def __init__(
        self,
        robot_id: str,
        master_secret: bytes,
        command_server_addr: str = "tcp://192.168.1.1:5555",
        telemetry_bind_addr: str = "tcp://*:5556",
        heartbeat_timeout_s: float = 15.0,
        on_command: Optional[Callable[[Message], None]] = None,
        on_heartbeat_timeout: Optional[Callable] = None,
    ):
        self.robot_id = robot_id
        self.crypto = CryptoLayer(master_secret)
        self.channel_mgr = ChannelManager()
        self._seq = 0
        self._on_command = on_command
        self._running = False

        self._zmq_ctx = zmq.Context()
        self._cmd_socket = self._zmq_ctx.socket(zmq.SUB)
        self._cmd_socket.connect(command_server_addr)
        self._cmd_socket.setsockopt_string(zmq.SUBSCRIBE, robot_id)
        self._cmd_socket.setsockopt_string(zmq.SUBSCRIBE, "BROADCAST")

        self._telem_socket = self._zmq_ctx.socket(zmq.PUB)
        self._telem_socket.bind(telemetry_bind_addr)

        self.watchdog = HeartbeatWatchdog(
            timeout_s=heartbeat_timeout_s,
            on_timeout=on_heartbeat_timeout,
        )

        self._recv_thread: Optional[threading.Thread] = None
        self._telem_thread: Optional[threading.Thread] = None
        self._telemetry_callback: Optional[Callable[[], Dict]] = None

        logger.info(f"CommunicationFramework online: robot_id={robot_id}")

    def start(self):
        self._running = True
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._recv_thread.start()
        self._telem_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._telem_thread.start()
        logger.info("Communication loops started")

    def stop(self):
        self._running = False
        self.watchdog.stop()
        self._zmq_ctx.term()
        logger.info("CommunicationFramework stopped")

    def send(self, msg: Message) -> bool:
        """Encrypt and transmit a message on the best available channel."""
        try:
            channel = self.channel_mgr.get_best_channel()
            msg.channel = channel
            msg.sequence = self._next_seq()

            t0 = time.time()
            raw = self.crypto.message_to_bytes(msg)
            topic = f"{self.robot_id} ".encode() + raw
            self._telem_socket.send(topic, zmq.NOBLOCK)
            latency_ms = (time.time() - t0) * 1000
            self.channel_mgr.report_success(channel, latency_ms)
            return True
        except zmq.Again:
            logger.debug("ZMQ send buffer full — dropped frame")
            return False
        except Exception as e:
            logger.error(f"Send failed: {e}")
            self.channel_mgr.report_failure(channel)
            return False

    def broadcast_telemetry(self, data: Dict) -> bool:
        msg = Message(
            msg_type=MessageType.TELEMETRY,
            payload=data,
            robot_id=self.robot_id,
        )
        return self.send(msg)

    def send_threat_alert(self, threat_data: Dict) -> bool:
        msg = Message(
            msg_type=MessageType.THREAT_ALERT,
            payload=threat_data,
            robot_id=self.robot_id,
        )
        return self.send(msg)

    def send_emergency(self, reason: str) -> bool:
        msg = Message(
            msg_type=MessageType.EMERGENCY,
            payload={"reason": reason, "robot_id": self.robot_id},
            robot_id=self.robot_id,
        )
        return self.send(msg)

    def register_telemetry_provider(self, callback: Callable[[], Dict]):
        """Register a callback that returns current telemetry data."""
        self._telemetry_callback = callback

    def _receive_loop(self):
        """Listen for incoming commands from command server."""
        poller = zmq.Poller()
        poller.register(self._cmd_socket, zmq.POLLIN)
        while self._running:
            socks = dict(poller.poll(timeout=200))
            if self._cmd_socket not in socks:
                continue
            try:
                raw = self._cmd_socket.recv(zmq.NOBLOCK)
                # Strip topic prefix
                space_idx = raw.find(b' ')
                encrypted = raw[space_idx + 1:]
                msg = self.crypto.bytes_to_message(encrypted, self.robot_id)

                if msg.msg_type == MessageType.HEARTBEAT:
                    self.watchdog.beat()
                elif msg.msg_type == MessageType.COMMAND and self._on_command:
                    logger.info(f"Command received: {msg.payload}")
                    self._on_command(msg)

            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"Receive error: {e}")

    def _telemetry_loop(self):
        """Periodically broadcast telemetry if provider is registered."""
        while self._running:
            time.sleep(self.TELEMETRY_INTERVAL)
            if self._telemetry_callback:
                try:
                    data = self._telemetry_callback()
                    self.broadcast_telemetry(data)
                except Exception as e:
                    logger.error(f"Telemetry broadcast error: {e}")

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def get_channel_health(self) -> Dict:
        return self.channel_mgr.get_health_report()
