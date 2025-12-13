#!/usr/bin/env python3
import json
import random
import socket
import sys
import threading

HOST = "127.0.0.1"
PORT = 2087


class MockLSPServer:
    """A minimal, extendable mock Language Server Protocol server.

    This class implements a small set of LSP handlers and provides
    placeholders for workspace attachment and diagnostic handlers.
    """

    def __init__(self):
        self.conversation_history = []
        # Simple in-memory store for attachments keyed by id
        self._attachments = {}
        self.responses = {
            "initialize": self.initialize,
            "textDocument/completion": self.completion,
            "workspace/attachmentPut": self.attachment_put,
            "workspace/attachmentGet": self.attachment_get,
            "workspace/diagnosticRequest": self.diagnostic_request,
        }

    def handle_client(self, conn, addr):
        """Handle a single client connection.

        This method reads LSP messages from the socket, validates them,
        dispatches to per-method handlers and writes responses.
        """
        try:
            print(f"Mock LSP: accepted connection from {addr}", file=sys.stderr)
            while True:
                msg = self.read_lsp_message(conn)
                if msg is None:
                    break

                # Basic validation
                if not self.is_valid_message(msg):
                    print(f"Mock LSP: invalid message: {msg}", file=sys.stderr)
                    # Ignore malformed messages rather than crashing
                    continue

                method = msg.get("method")
                params = msg.get("params")
                response = None

                try:
                    if method in self.responses:
                        response = self.responses[method](msg, params)
                    else:
                        # Default: echo back an empty result for requests with id
                        if "id" in msg:
                            response = {
                                "jsonrpc": "2.0",
                                "id": msg["id"],
                                "result": None,
                            }
                        else:
                            # Notification: no response required
                            response = None
                except KeyError as e:
                    # handler expected fields missing
                    print(f"Handler KeyError for {method}: {e}", file=sys.stderr)
                    if "id" in msg:
                        response = {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {"code": -32602, "message": str(e)},
                        }
                except Exception as e:
                    # Generic handler failure
                    print(f"Handler error for {method}: {e}", file=sys.stderr)
                    if "id" in msg:
                        response = {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {"code": -32603, "message": str(e)},
                        }

                if response is not None:
                    print(
                        f"Mock LSP: sending response for {method} with id {msg.get('id')}",
                        file=sys.stderr,
                    )
                    try:
                        self.send_lsp_message(conn, response)
                    except OSError as e:
                        print(f"Failed to send response: {e}", file=sys.stderr)
                        break
        except ConnectionResetError:
            print("Connection reset by peer", file=sys.stderr)
        except OSError as e:
            print(f"Socket error: {e}", file=sys.stderr)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def read_lsp_message(self, conn):
        """Read a single LSP message from `conn` and return the parsed JSON.

        Returns None on EOF or on parse failure.
        """
        header = b""
        try:
            while b"\r\n\r\n" not in header:
                ch = conn.recv(1)
                if not ch:
                    return None
                header += ch
        except ConnectionResetError:
            return None

        header_str, _, _ = header.partition(b"\r\n\r\n")
        headers = header_str.decode("utf-8", errors="replace").split("\r\n")
        content_length = 0
        for h in headers:
            if h.lower().startswith("content-length:"):
                try:
                    content_length = int(h.split(":", 1)[1].strip())
                except ValueError:
                    print(f"Invalid Content-Length header: {h}", file=sys.stderr)
                    return None

        body = b""
        try:
            while len(body) < content_length:
                chunk = conn.recv(content_length - len(body))
                if not chunk:
                    return None
                body += chunk
        except ConnectionResetError:
            return None

        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON body: {e}", file=sys.stderr)
            return None

    def send_lsp_message(self, conn, obj):
        data = json.dumps(obj, separators=(",", ":"))
        header = f"Content-Length: {len(data.encode('utf-8'))}\r\n\r\n"
        conn.sendall(header.encode("utf-8") + data.encode("utf-8"))

    def is_valid_message(self, msg):
        """Perform minimal validation on an incoming LSP message.

        We check that the message is a dict and that `jsonrpc` is 2.0.
        More checks can be added as needed.
        """
        if not isinstance(msg, dict):
            return False
        if msg.get("jsonrpc") != "2.0":
            return False
        # method may be missing for responses; accept those too
        return True

    def initialize(self, msg, params):
        # Handle initialize method
        resp = {"jsonrpc": "2.0", "id": msg.get("id"), "result": {"capabilities": {}}}
        return resp

    def completion(self, msg, params):
        # Handle textDocument/completion method
        resp = {"jsonrpc": "2.0", "id": msg.get("id"), "result": []}
        return resp

    def attachment_put(self, msg, params):
        """Handle workspace/attachmentPut.

        This stores attachments in a small in-memory dict. Real LSP
        implementations may stream bytes or store to disk.
        """
        if not isinstance(params, dict) or "id" not in params:
            raise KeyError("params.id required for attachmentPut")
        attach_id = params["id"]
        content = params.get("content")
        self._attachments[attach_id] = content
        return {"jsonrpc": "2.0", "id": msg.get("id"), "result": None}

    def attachment_get(self, msg, params):
        """Handle workspace/attachmentGet.

        Returns previously stored attachment or a not-found error.
        """
        if not isinstance(params, dict) or "id" not in params:
            raise KeyError("params.id required for attachmentGet")
        attach_id = params["id"]
        if attach_id not in self._attachments:
            return {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "error": {"code": -32000, "message": "attachment not found"},
            }
        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {"content": self._attachments[attach_id]},
        }

    def diagnostic_request(self, msg, params):
        """Placeholder for a diagnostics request. Returns an empty diagnostics list.

        In a real server this would run analysis and return any findings.
        """
        return {"jsonrpc": "2.0", "id": msg.get("id"), "result": {"diagnostics": []}}


class UnbreakableOracle:
    def __init__(self):
        self.conversation_history = []
        self.responses = {
            "open": self.open_conversation,
            "close": self.close_conversation,
            "help": self.show_help,
        }

    def open_conversation(self):
        print("Unbreakable Oracle is now open for business!")
        while True:
            user_input = input()
            if user_input in ["quit", "exit"]:
                break
            response = self.generate_response(user_input)
            print(response)

    def close_conversation(self):
        print("Conversation closed.")
        exit()

    def show_help(self):
        print("Available commands:")
        print("* open: Start a new conversation")
        print("* close: End the conversation")
        print("* help: Display this menu")

    def generate_response(self, user_input):
        # Use NLP techniques to analyze user input
        # For simplicity, let's use a simple keyword-based approach
        keywords = ["conversational", "topic", "question"]
        if any(keyword in user_input.lower() for keyword in keywords):
            return self.conversational_response(user_input)
        else:
            return "I didn't quite understand that."

    def conversational_response(self, user_input):
        # Use a simple AI algorithm to generate a response
        # For simplicity, let's use a random selection from a list
        responses = [
            "That's an interesting question!",
            "I'm not sure I understand.",
            "Let me try to help you with that.",
        ]
        return random.choice(responses)


def run():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"Mock LSP server listening on {HOST}:{PORT}", file=sys.stderr)
    try:
        server = MockLSPServer()
        while True:
            conn, addr = s.accept()
            # reuse the same server instance so attachments/diagnostics persist
            t = threading.Thread(
                target=server.handle_client, args=(conn, addr), daemon=True
            )
            t.start()
    except KeyboardInterrupt:
        s.close()


run()
