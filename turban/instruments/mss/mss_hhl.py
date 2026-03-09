import logging
import sys
import numpy as np


# Setup logging module
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class hhl:
    """A processor for the HHL binary datastream from Sea & Sun Technology."""

    def __init__(self, verbosity=logging.DEBUG, config={}):
        self.logger = logging.getLogger("turban.instruments.mss_hhl")
        self.logger.setLevel(verbosity)
        self.config = config
        self.buffer = b""  # a binary buffer for the rawdata
        self.ngood = 0
        self.nbad = 0

    def add_to_buffer(self, data):
        """Append data to the internal buffer.

        Parameters
        ----------
        data : bytes
            Data to append to the buffer.
        """
        self.buffer += data

    def align_buffer(self):
        """Align buffer to channel 0.

        Returns
        -------
        None or bool
            None if alignment failed, True if buffer is valid and ready to process.
        """

        funcname = __name__ + ".align_buffer():"
        self.FLAG_VALID_HHL = False
        self.FLAG_VALID_BUFFER = False

        # check for a valid HHL packet first, if found loop over packets and seek for channel 0, if found check if buffer is still enough to create one datapacket
        # Seek for a start, find two "H" and one "L" pattern
        n = len(self.buffer)
        print(funcname + "Processing length", n)
        if n >= 48:  # We need at least 3*16 bytes for one complete datapacket
            # for i in range(3):
            for i in range(16):  # Check 16 bytes
                data_tmp = self.decode_HHL(self.buffer)
                # Check if the datastream is valid, if not delete the first byte and try again
                if data_tmp is not None:
                    print("Found valid HHL packet after", i)
                    self.FLAG_VALID_HHL = True
                    channel = data_tmp[0]
                    data = data_tmp[1]
                    break
                else:
                    # print('Did not found valid HHL packet')
                    self.buffer = self.buffer[1:]

            print("Valid", self.FLAG_VALID_HHL)
            # Found valid HHL, now search for channel 0
            if self.FLAG_VALID_HHL == False:
                print("No valid hhl")
                return None
            else:
                # for i in range(0,16):
                for i in range(0, 20):
                    if channel == 0:  # Channel 0, great!
                        print("Found channel 0")
                        n = len(self.buffer)
                        if (
                            n >= 48
                        ):  # We need at least 3*16 bytes for one complete datapacket
                            print("Enough data to decode")
                            self.FLAG_VALID_BUFFER = True
                            return True
                        # print('Found channel 0')
                        break
                    else:  # throw data away and check next three bytes
                        self.buffer = self.buffer[3:]
                        [channel, data] = self.decode_HHL(self.buffer)

    def process_buffer(self):
        """Process data in the buffer.

        Returns
        -------
        list of (ndarray, ndarray) or None
            [channel_cat, data_cat] where channel_cat is an ndarray of channel
            numbers and data_cat is an ndarray of shape (n_packets, 16) with
            decoded data values. Returns None if buffer cannot be aligned.
        """
        funcname = __name__ + ".process_buffer():"
        self.FLAG_VALID_HHL = True  # legacy
        self.FLAG_VALID_BUFFER = True  # legacy
        nbad = 0

        align = self.align_buffer()
        if align is None:
            self.logger.debug("Could not align buffer")
            return None
        else:
            # If we have a valid buffer, starts with channel 0 and has at least 48 bytes
            # if self.FLAG_VALID_BUFFER:
            if True:
                print(funcname + " Processing rawdata")
                data_cat = []
                channel_cat = []
                while True:
                    if len(self.buffer) < 48:  # We need at least 3 bytes to process
                        print(funcname + " Not enough data left")
                        break
                    else:  # Process data
                        data_tmp = []
                        for i in range(0, 16):
                            # print(funcname,i,len(self.buffer))
                            data_proc = self.decode_HHL(self.buffer[0:3])
                            if data_proc is not None:
                                channel = data_proc[0]
                                data = data_proc[1]
                            else:
                                print("Problem decoding, realigning", data_proc)
                                align = self.align_buffer()
                                if align is None:
                                    self.logger.debug("Could not realign align buffer")
                                    return None

                            self.buffer = self.buffer[3:]
                            channel_cat.append(channel)
                            if channel == i:
                                data_tmp.append(data)
                            else:
                                print("Bad channel", channel, i)
                                align = self.align_buffer()
                                if align is None:
                                    self.logger.debug("Could not realign align buffer")
                                    return None
                                # return None

                            # print('Channel',channel)
                            # print('data {:04x}'.format(data))

                        # print('Len',len(data_tmp))
                        if len(data_tmp) == 16:
                            data_cat.append(data_tmp)
                            # print('Data tmp',data_tmp)
                            self.ngood += 1
                        else:
                            self.nbad += 1
                channel_cat = np.asarray(channel_cat)
                data_cat = np.asarray(data_cat)
                data_return = [channel_cat, data_cat]
                print("N Good, N Bad hhl", self.ngood, self.nbad)
                return data_return

    def process_buffer_legacy(self):
        """Process data in the buffer (legacy version).

        Returns
        -------
        list of (ndarray, ndarray) or None
            [channel_cat, data_cat] where channel_cat is an ndarray of channel
            numbers and data_cat is an ndarray of shape (n_packets, 16) with
            decoded data values. Returns None if buffer cannot be aligned.
        """
        funcname = __name__ + ".process_buffer():"
        self.FLAG_VALID_HHL = False
        self.FLAG_VALID_BUFFER = False

        # check for a valid HHL packet first, if found loop over packets and seek for channel 0, if found check if buffer is still enough to create one datapacket
        # Seek for a start, find two "H" and one "L" pattern
        n = len(self.buffer)
        print(funcname + "Processing length", n)
        if n >= 48:  # We need at least 3*16 bytes for one complete datapacket
            for i in range(3):
                data_tmp = self.decode_HHL(self.buffer)
                # Check if the datastream is valid, if not delete the first byte and try again
                if data_tmp is not None:
                    # print('Found valid HHL packet')
                    self.FLAG_VALID_HHL = True
                    channel = data_tmp[0]
                    data = data_tmp[1]
                    break
                else:
                    # print('Did not found valid HHL packet')
                    self.buffer = self.buffer[1:]

            print("Valid", self.FLAG_VALID_HHL)
            # Found valid HHL, now search for channel 0
            if self.FLAG_VALID_HHL == False:
                print("No valid hhl")
            else:
                # for i in range(0,16):
                for i in range(0, 20):
                    if channel == 0:  # Channel 0, great!
                        # print('Found channel 0')
                        n = len(self.buffer)
                        if (
                            n >= 48
                        ):  # We need at least 3*16 bytes for one complete datapacket
                            print("Enough data to decode")
                            self.FLAG_VALID_BUFFER = True
                        # print('Found channel 0')
                        break
                    else:  # throw data away and check next three bytes
                        self.buffer = self.buffer[3:]
                        [channel, data] = self.decode_HHL(self.buffer)

            # If we have a valid buffer, starts with channel 0 and has at least 48 bytes
            if self.FLAG_VALID_BUFFER:
                print(funcname + " Processing rawdata")
                data_cat = []
                channel_cat = []
                while True:
                    if len(self.buffer) < 48:  # We need at least 3 bytes to process
                        print(funcname + " Not enough data left")
                        break
                    else:  # Process data
                        data_tmp = []
                        for i in range(0, 16):
                            print(funcname, i, len(self.buffer))
                            data_proc = self.decode_HHL(self.buffer[0:3])
                            if data_proc is not None:
                                channel = data_proc[0]
                                data = data_proc[1]
                            else:
                                print("Problem decoding", data_proc)
                                return None
                            self.buffer = self.buffer[3:]
                            channel_cat.append(channel)
                            if channel == i:
                                data_tmp.append(data)
                            else:
                                print("Bad channel", channel, i)
                                return None

                            # print('Channel',channel)
                            # print('data {:04x}'.format(data))

                        data_cat.append(data_tmp)
                channel_cat = np.asarray(channel_cat)
                data_cat = np.asarray(data_cat)
                data_return = [channel_cat, data_cat]
                # print('Data hhl',data_cat[-1])
                return data_return

    def decode_HHL(self, hhldata):
        """Decode three bytes of HHL data into channel and value.

        Parameters
        ----------
        hhldata : bytes
            HHL packet data, length >= 3.

        Returns
        -------
        list of (int, int) or None
            [channel, data] where channel is the channel number and data is
            the decoded 16-bit value. Returns None if the packet is invalid.
        """
        # Check if its a valid packet
        if len(hhldata) >= 2:
            # print('data',data,data[0:1])
            FLAG0 = hhldata[0] & 0x01 == 1
            FLAG1 = hhldata[1] & 0x01 == 1
            FLAG2 = hhldata[2] & 0x01 == 0
            if FLAG0 and FLAG1 and FLAG2:
                pass
            else:
                return None
        else:
            return None

        HHL0 = hhldata[0]
        HHL1 = hhldata[1]
        HHL2 = hhldata[2]
        # print("HHL: {:2x} {:2x} {:2x}".format(HHL0, HHL1, HHL2))
        channel = HHL2 >> 3
        data = HHL0 >> 1
        data = data | ((HHL1 & 0xFE) << 6)
        data = data | ((HHL2 & 0x06) << 13)
        return [channel, data]

    def valid_packet(self, data):
        """Check if the data packet has a valid HHL pattern.

        Parameters
        ----------
        data : bytes
            Data to validate.

        Returns
        -------
        bool
            True if the first three bytes have the HHL pattern, False otherwise.
        """
        if len(data) >= 2:
            # print('data',data,data[0:1])
            FLAG0 = data[0] & 0x01 == 1
            FLAG1 = data[1] & 0x01 == 1
            FLAG2 = data[2] & 0x01 == 0
            if FLAG0 and FLAG1 and FLAG2:
                return True
            else:
                return False
        else:
            return False
