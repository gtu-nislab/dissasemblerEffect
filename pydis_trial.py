import pydis

instructions = b'\x51\x8d\x45\xff\x50\xff\x75\x0c\xff\x75\x08\xff\x15\xa0\xa5\x48\x76\x85\xc0\x0f\x88\xfc\xda\x02\x00'
instruction_pointer = 0x007FFFFFFF400000

for instruction in pydis.decode(instructions, instruction_pointer):
    print(instruction)
