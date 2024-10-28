import numpy as np
import sys

R, I, S, B, J, U = 0, 1, 2, 3, 4, 5


fmts = ["R", "I", "S", "B", "J", "U"]

opcodes = {
    0b0110011: R,
    0b0010011: I,
    0b0000011: I,
    0b0100011: S,
    0b1100011: B,
    0b1101111: J,
    0b1100111: I,
    0b0110111: U,
    0b0010111: U,
    0b1110011: I,
}

opcode_instructions = {
    # R-Type RV32I + RV32M
    0b0110011: {
        0x0: {
            0x00: "add",
            0x20: "sub",
            0x01: "mul"
        },
        0x4: {
            0x00: "xor",
            0x01: "div"
        },
        0x6: {
            0x00: "or",
            0x01: "rem"
        },
        0x7: {
            0x00: "and",
            0x01: "remu"
        },
        0x1: {
            0x00: "sll",
            0x01: "mulh"
        },
        0x5: {
            0x00: "srl",
            0x20: "sra",
            0x01: "divu"
        },
        0x2: {
            0x00: "slt",
            0x01: "mulsu"
        },
        0x3: {
            0x00: "sltu",
            0x01: "mulu"
        }
    },
    # I-type RV32I
    0b0010011: {
        0x0: "addi",
        0x4: "xori",
        0x6: "ori",
        0x7: "andi",
        0x1: "slli",
        0x5: {
            0x00: "srli",
            0x20: "srai",
        },
        0x2: "slti",
        0x3: "sltiu"
    },
    # I-type RV32I Load
    0b0000011: {
        0x0: "lb",
        0x1: "lh",
        0x2: "lw",
        0x4: "lbu",
        0x5: "lhu"
    },
    # S-type RV32I Store
    0b0100011: {
        0x0: "sb",
        0x1: "sh",
        0x2: "sw"
    },
    # B-type branch
    0b1100011: {
        0x0: "beq",
        0x1: "bne",
        0x4: "blt",
        0x5: "bge",
        0x6: "bltu",
        0x7: "bgeu"
    },
    # J-type jal
    0b1101111: "jal",
    0b1100111: {
        0x0: "jalr"
    },
    0b0110111: "lui",
    0b0010111: "auipc",
    0b1110011: {
        0x0: {
            0x0: "ecall",
            0x1: "ebreak"
        }
    },
}


class reg:
    def __init__(self, addr) -> None:
        if addr < 0 or addr > 32:
            raise ValueError("Register address cannot be >32")
        self.address = addr

    def __str__(self) -> str:
        return f"reg x{self.address}"


class instruction:
    def __init__(self, inst_bytes) -> None:
        hexint = inst_bytes
        #print(inst_bytes, hexint, bin(hexint), hex(hexint))
        self.opcode = self.fmt = self.funct3 = self.funct7 = self.imm = self.rd = self.rs1 = self.rs2 = None
        self.opcode = hexint & 0x7f

        if self.opcode in opcodes:
            self.fmt = opcodes[self.opcode]
            if self.fmt == R:
                self.rd = reg(hexint >> 7 & 0x1f)
                self.funct3 = hexint >> 12 & 0x7
                self.rs1 = reg(hexint >> 15 & 0x1f)
                self.rs2 = reg(hexint >> 20 & 0x1f)
                self.funct7 = hexint >> 25
            elif self.fmt == I:
                self.rd = reg(hexint >> 7 & 0x1f)
                self.funct3 = hexint >> 12 & 0x7
                self.rs1 = reg(hexint >> 15 & 0x1f)
                # https://en.wikipedia.org/wiki/Two's_complement
                if hexint >> 31:
                    self.imm = - (1+(0x7ff - ((hexint >> 20) & 0x7ff)))
                else:
                    self.imm = hexint >> 20

                self.funct7 = (hexint & 0xfe000000) >> 25
            elif self.fmt == B:
                self.funct3 = hexint >> 12 & 0x7
                self.rs1 = reg(hexint >> 15 & 0x1f)
                self.rs2 = reg(hexint >> 20 & 0x1f)
                self.imm = ((hexint & 0x80000000) >> 19) | ((hexint & 0x80) << 4) | (
                    (hexint >> 20) & 0x7e0) | ((hexint >> 7) & 0x1e)  # hexint >> 25
                if hexint >> 31:
                    self.imm = -(2+(0x1ffe-(self.imm & 0x1ffe)))
            elif self.fmt == J:
                self.rd = reg(hexint >> 7 & 0x1f)
                self.imm = hexint >> 12
            elif self.fmt == U:
                self.rd = reg(hexint >> 7 & 0x1f)
                self.imm = hexint >> 12
            elif self.fmt == S:
                self.imm = ((hexint & 0xfe000000) >> 20) | (
                    (hexint >> 7) & 0x1f)
                self.funct3 = hexint >> 12 & 0x7
                self.rs1 = reg(hexint >> 15 & 0x1f)
                self.rs2 = reg(hexint >> 20 & 0x1f)

        else:
            raise ValueError("Unknown opcode", bin(self.opcode))

    def solve_instruction(self) -> str:
        try:
            inst = opcode_instructions[self.opcode]
            if type(inst) == dict:
                if self.funct3 != None:
                    inst = inst[self.funct3]

            if (type(inst) == dict) and (self.funct7 != None):
                inst = inst[self.funct7]

            if type(inst) == dict:
                print("unk", inst, bin(self.opcode),
                      self.funct3, self.funct7, self.imm)
                inst = "unk"
        except Exception as e:
            print("err Cannot solve instruction", e, bin(
                self.opcode), self.funct3, self.funct7)
            inst = "err"
        return inst

    def disassemble(self) -> str:
        disassembled = self.solve_instruction()
        if self.fmt == I:
            disassembled += " "+", ".join(
                list(map(lambda x: "None" if (x == None) else "x"+str(x), [self.rd.address, self.rs1.address])))
            disassembled += ", "+", ".join(
                list(map(lambda x: "None" if (x == None) else str(x), [self.imm])))
        elif self.fmt == R:
            disassembled += " "+", ".join(
                list(map(lambda x: "None" if (x == None) else "x"+str(x), [self.rd.address, self.rs1.address, self.rs2.address])))
        elif self.fmt == J:
            disassembled += " "+", ".join(
                list(map(lambda x: "None" if (x == None) else "x"+str(x), [self.rd.address])))
            disassembled += ", "+", ".join(
                list(map(lambda x: "None" if (x == None) else str(x), [self.imm])))
        elif self.fmt == U:
            disassembled += " "+", ".join(
                list(map(lambda x: "None" if (x == None) else "x"+str(x), [self.rd.address])))
            disassembled += ", "+", ".join(
                list(map(lambda x: "None" if (x == None) else str(x), [self.imm])))
        elif self.fmt == B:
            disassembled += " "+", ".join(
                list(map(lambda x: "None" if (x == None) else "x"+str(x), [self.rs1.address, self.rs2.address])))
            disassembled += ", "+", ".join(
                list(map(lambda x: "None" if (x == None) else str(x), [self.imm])))
        elif self.fmt == S:
            disassembled += " "+", ".join(
                list(map(lambda x: "None" if (x == None) else "x"+str(x), [self.rs1.address, self.rs2.address])))
            disassembled += ", "+", ".join(
                list(map(lambda x: "None" if (x == None) else str(x), [self.imm])))
        return disassembled

    def __str__(self) -> str:
        return f"{bin(self.opcode)}, {self.solve_instruction()} {fmts[self.fmt]} rd:{self.rd}, funct3:{self.funct3}, rs1:{self.rs1}, rs2:{self.rs2}, imm:{self.imm}, funct7:{self.funct7}"

# Implementing a subset of RV32I
class cpu:
    def __init__(self, dram_initial=np.zeros(1024, dtype=np.byte), startoffset=0) -> None:
        self.regs = np.zeros(32, dtype=np.int32)
        self.startoffset = startoffset
        self.dram = dram_initial
        self.pc = startoffset
        # x1 = -1 kun ei hyppyjä tehty
        self.regs[1] = -1
        self.stdio = []

    def fakex0(self):
        self.regs[0] = 0

    def dumpregs(self):
        x = 0
        for i in self.regs:
            print(f"{("x"+str(x)).ljust(3)}:{'{0:032b}'.format(i)} = {i}")
            x += 1

    def _addi(self, rd, rs1, imm):
        self.regs[rd.address] = self.regs[rs1.address]+imm

    def _add(self, rd, rs1, rs2):
        self.regs[rd.address] = self.regs[rs1.address]+self.regs[rs2.address]

    def _sub(self, rd, rs1, rs2):
        self.regs[rd.address] = self.regs[rs1.address]-self.regs[rs2.address]

    def _or(self, rd, rs1, rs2):
        self.regs[rd.address] = self.regs[rs1.address] | self.regs[rs2.address]

    def _and(self, rd, rs1, rs2):
        self.regs[rd.address] = self.regs[rs1.address] & self.regs[rs2.address]

    def _xor(self, rd, rs1, rs2):
        self.regs[rd.address] = self.regs[rs1.address] ^ self.regs[rs2.address]

    def _div(self, rd, rs1, rs2):
        # TODO
        self.regs[rd.address] = self.regs[rs1.address] / self.regs[rs2.address]

    def _divu(self, rd, rs1, rs2):
        # TODO
        self.regs[rd.address] = abs(
            self.regs[rs1.address]) / abs(self.regs[rs2.address])

    def _rem(self, rd, rs1, rs2):
        # TODO
        self.regs[rd.address] = self.regs[rs1.address] % self.regs[rs2.address]

    def _remu(self, rd, rs1, rs2):
        # TODO
        self.regs[rd.address] = abs(
            self.regs[rs1.address]) % abs(self.regs[rs2.address])

    def _mul(self, rd, rs1, rs2):
        # TODO
        self.regs[rd.address] = self.regs[rs1.address] * self.regs[rs2.address]

    def _mulu(self, rd, rs1, rs2):
        # TODO
        self.mul(rd, rs1, rs2)

    def _mulsu(self, rd, rs1, rs2):
        # TODO
        self.mul(rd, rs1, rs2)

    def _mulh(self, rd, rs1, rs2):
        # TODO
        self.mul(rd, rs1, rs2)

    def _jal(self, rd, imm):
        self.regs[rd.address] = self.pc + 4
        self.pc = self.pc + imm + 4

    def _jalr(self, rd, rs1, imm):
        self.regs[rd.address] = self.pc + 4
        self.pc = self.regs[rs1.address]+imm

    def _sll(self, rd, rs1, rs2):
        self.regs[rd.address] = self.regs[rs1.address] << self.regs[rs2.address]

    def _srl(self, rd, rs1, rs2):
        self.regs[rd.address] = self.regs[rs1.address] >> self.regs[rs2.address]

    def _sra(self, rd, rs1, rs2):
        # TODO
        self.regs[rd.address] = self.regs[rs1.address] >> self.regs[rs2.address]

    def _slli(self, rd, rs1, imm):
        self.regs[rd.address] = self.regs[rs1.address] << imm

    def _srli(self, rd, rs1, imm):
        self.regs[rd.address] = self.regs[rs1.address] >> imm

    def _srai(self, rd, rs1, imm):
        # TODO
        self.regs[rd.address] = self.regs[rs1.address] >> imm

    def _ecall(self):
        call = self.regs[17]
        args = self.regs[10]
        match call:
            case 11:
                print("[ECALL] 11:printchar", chr(args))
                self.stdio.append(chr(args))
            case 5:
                print("[ECALL] 5:printint", args)
                self.regs[10] = np.int32(input(">"))
            case 4:
                # printstring
                # removed, depends on debugger
                pass
    
    def _ebreak(self):
        print("breakpoint")
        # removed, depends on debugger

    def _beq(self, rs1, rs2, imm):
        if self.regs[rs1.address] == self.regs[rs2.address]:
            self.pc += imm 
            self.pc -= 4 

    def _bne(self, rs1, rs2, imm):
        if self.regs[rs1.address] != self.regs[rs2.address]:
            self.pc += imm 
            self.pc -= 4 

    def _blt(self, rs1, rs2, imm):
        if self.regs[rs1.address] < self.regs[rs2.address]:
            self.pc += imm 
            self.pc -= 4 

    def _bge(self, rs1, rs2, imm):
        if self.regs[rs1.address] >= self.regs[rs2.address]:
            self.pc += imm 
            self.pc -= 4 

    def _bltu(self, rs1, rs2, imm):
        if abs(self.regs[rs1.address]) < abs(self.regs[rs2.address]):
            self.pc += imm 
            self.pc -= 4 

    def _bgeu(self, rs1, rs2, imm):
        if abs(self.regs[rs1.address]) >= abs(self.regs[rs2.address]):
            self.pc += imm 
            self.pc -= 4 

    def _xori(self, rd, rs1, imm):
        self.regs[rd.address] = self.regs[rs1.address] ^ imm

    def _ori(self, rd, rs1, imm):
        self.regs[rd.address] = self.regs[rs1.address] | imm

    def _andi(self, rd, rs1, imm):
        self.regs[rd.address] = self.regs[rs1.address] & imm

    def _slti(self, rd, rs1, imm):
        self.regs[rd.address] = int(self.regs[rs1.address] < imm)

    def _sltiu(self, rd, rs1, imm):
        self.regs[rd.address] = int(abs(self.regs[rs1.address]) < abs(imm))

    def _slt(self, rd, rs1, rs2):
        self.regs[rd.address] = int(
            self.regs[rs1.address] < self.regs[rs2.address])

    def _sltu(self, rd, rs1, rs2):
        self.regs[rd.address] = int(
            abs(self.regs[rs1.address]) < abs(self.regs[rs2.address]))

    def _lb(self, rd, rs1, imm):
        # TODO IMPLEMENT
        print(f"[DRAM] read {self.regs[rs1.address] +
              imm}, with value of {self.dram[self.regs[rs1.address]+imm]}")
        self.regs[rd.address] = self.dram[self.regs[rs1.address]+imm]

    def _lh(self, rd, rs1, imm):
        # TODO IMPLEMENT

        print(f"[DRAM] read {self.regs[rs1.address] +
              imm}, with value of {self.dram[self.regs[rs1.address]+imm]}")
        self.regs[rd.address] = self.dram[self.regs[rs1.address] +
                                          imm] + self.dram[self.regs[rs1.address]+imm+1]

    def _lw(self, rd, rs1, imm):
        # TODO IMPLEMENT

        print(f"[DRAM] read {self.regs[rs1.address] +
              imm}, with value of {self.dram[self.regs[rs1.address]+imm]}")
        self.regs[rd.address] = self.dram[self.regs[rs1.address] +
                                          imm] + self.dram[self.regs[rs1.address]+imm+1] + self.dram[self.regs[rs1.address]+imm+2]+self.dram[self.regs[rs1.address]+imm+14]
    def _lbu(self, rd, rs1, imm):
        print(f"[DRAM] read {self.regs[rs1.address] +
              imm}, with value of {self.dram[self.regs[rs1.address]+imm]}")
        self.regs[rd.address] = self.dram[self.regs[rs1.address]+imm]

    def _lhu(self, rd, rs1, imm):
        print(f"[DRAM] read {self.regs[rs1.address] +
              imm}, with value of {self.dram[self.regs[rs1.address]+imm]}")
        self.regs[rd.address] = self.dram[self.regs[rs1.address]+imm]+self.dram[self.regs[rs1.address]+imm+1]

    # TODO Implement dram store
    def _sb(self, rs1, rs2, imm):
        pass

    def _sh(self, rs1, rs2, imm):
        pass

    def _sw(self, rs1, rs2, imm):
        pass

    def _lui(self, rd, imm):
        # TODO
        self.regs[rd.address] = imm << 12
    
    def _auipc(self, rd, imm):
        # TODO
        self.regs[rd.address] = self.pc + (imm << 12)


    def getInstAtOffset(self, address, ramoffset=0):
        instbin = parseBinaryInst(
            self.dram[ramoffset+(address):ramoffset+((address+4))])
        hexstring = "".join(
            list(map(lambda x: hex(x).split("x")[-1].zfill(2), instbin)))
        return int(hexstring, 16)

    # decode from dram
    def decode(self):
        inst = self.getInstAtOffset(self.pc)
        if inst == 0:
            # removed, depends on debugger
            print("\nPC does not refer to instructions, only zeros. Halting cpu!\n")
            return False
        else:
            # removed, depends on debugger
            pass
        return instruction(inst)

    # The CPU itself - a giant match statement
    def execute(self, inst: instruction):
        self.fakex0()
        print(f"[CPU-EXEC] PC:0x{hex(self.pc).split("x")[-1].upper()}")
        print(f"[CPU-EXEC] {inst}")
        match inst.opcode:
            case 0b0110011:
                match inst.funct3:
                    case 0x0:
                        match inst.funct7:
                            case 0x00:
                                self._add(inst.rd, inst.rs1, inst.rs2)
                            case 0x20:
                                self._sub(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._mul(inst.rd, inst.rs1, inst.rs2)
                    case 0x4:
                        match inst.funct7:
                            case 0x00:
                                self._xor(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._div(inst.rd, inst.rs1, inst.rs2)
                    case 0x6:
                        match inst.funct7:
                            case 0x00:
                                self._or(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._rem(inst.rd, inst.rs1, inst.rs2)
                    case 0x7:
                        match inst.funct7:
                            case 0x00:
                                self._and(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._remu(inst.rd, inst.rs1, inst.rs2)
                    case 0x1:
                        match inst.funct7:
                            case 0x00:
                                self._sll(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._mulh(inst.rd, inst.rs1, inst.rs2)
                    case 0x5:
                        match inst.funct7:
                            case 0x00:
                                self._srl(inst.rd, inst.rs1, inst.rs2)
                            case 0x20:
                                self._sra(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._divu(inst.rd, inst.rs1, inst.rs2)
                    case 0x2:
                        match inst.funct7:
                            case 0x00:
                                self._slt(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._mulsu(inst.rd, inst.rs1, inst.rs2)
                    case 0x3:
                        match inst.funct7:
                            case 0x00:
                                self._sltu(inst.rd, inst.rs1, inst.rs2)
                            case 0x01:
                                self._mulu(inst.rd, inst.rs1, inst.rs2)

            case 0b0010011:
                match inst.funct3:
                    case 0x0:
                        self._addi(inst.rd, inst.rs1, inst.imm)
                    case 0x4:
                        self._xori(inst.rd, inst.rs1, inst.imm)
                    case 0x6:
                        self._ori(inst.rd, inst.rs1, inst.imm)
                    case 0x7:
                        self._andi(inst.rd, inst.rs1, inst.imm)
                    case 0x1:
                        self._slli(inst.rd, inst.rs1, inst.imm)
                    case 0x5:
                        match inst.funct7:
                            case 0x00:
                                self._srli(inst.rd, inst.rs1, inst.imm)
                            case 0x20:
                                self._srai(inst.rd, inst.rs1, inst.imm)
                    case 0x2:
                        self._slti(inst.rd, inst.rs1, inst.imm)
                    case 0x3:
                        self._sltiu(inst.rd, inst.rs1, inst.imm)
            case 0b0000011:
                match inst.funct3:
                    case 0x0:
                        self._lb(inst.rd, inst.rs1, inst.imm)
                    case 0x1:
                        self._lh(inst.rd, inst.rs1, inst.imm)
                    case 0x2:
                        self._lw(inst.rd, inst.rs1, inst.imm)
                    case 0x4:
                        self._lbu(inst.rd, inst.rs1, inst.imm)
                    case 0x5:
                        self._lhu(inst.rd, inst.rs1, inst.imm)
            case 0b0100011:
                match inst.funct7:
                    case 0x0:
                        self._sb(inst.rs1, inst.rs2, inst.imm)
                    case 0x1:
                        self._sh(inst.rs1, inst.rs2, inst.imm)
                    case 0x2:
                        self._sw(inst.rs1, inst.rs2, inst.imm)
            case 0b1100011:
                match inst.funct3:
                    case 0x0:
                        self._beq(inst.rs1, inst.rs2, inst.imm)
                    case 0x1:
                        self._bne(inst.rs1, inst.rs2, inst.imm)
                    case 0x4:
                        self._blt(inst.rs1, inst.rs2, inst.imm)
                    case 0x5:
                        self._bge(inst.rs1, inst.rs2, inst.imm)
                    case 0x6:
                        self._bltu(inst.rs1, inst.rs2, inst.imm)
                    case 0x7:
                        self._bgeu(inst.rs1, inst.rs2, inst.imm)
            case 0b1101111:
                self._jal(inst.rd, inst.imm)
                self.pc -= 4
            case 0b1100111:
                match inst.funct3:
                    case 0x0:
                        self._jalr(inst.rd, inst.rs1, inst.imm)
            case 0b0110111:
                self._lui(inst.rd, inst.imm)
            case 0b0010111:
                self._auipc(inst.rd, inst.imm)
            case 0b1110011:
                match inst.funct7:
                    case 0x0:
                        self._ecall()
                    case 0x1:
                        self._ebreak()
        self.pc += 4


def loadFile(file):
    b = list(file.read())
    return b

def parseBinaryInst(binary):
    insts = []
    cinst = []
    if len(sys.argv) >= 3:
        if sys.argv[2] == "direct":
            print("Directly loading:", binary)
            return binary
    for i in binary:
        cinst.append(i)
        if len(cinst) == 4:
            insts = list(reversed(cinst))
    return insts

def getInstAtOffset(address, dram, ramoffset=0):
    instbin = parseBinaryInst(
        dram[ramoffset+(address*4):ramoffset+((address+1)*4)])
    hexstring = "".join(
        list(map(lambda x: hex(x).split("x")[-1].zfill(2), instbin)))
    return int(hexstring, 16)

def bytedump(fn, barr):
    f = open(fn, "wb")
    f.write(barr)
    f.close()

dram = bytearray(4096)
loaded_data = loadFile(open(sys.argv[1], "rb"))
dram[0:len(loaded_data)] = loaded_data

sim_cpu = cpu(dram_initial=dram, startoffset=0)

while True:
    dec_inst = sim_cpu.decode()
    if dec_inst == False:
        # removed, depends on debugger
        break
    sim_cpu.execute(dec_inst)

sim_cpu.dumpregs()
