// SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract ERC721D {
    mapping(uint256 => address) public ownerOf;
    mapping(uint256 => string) public spanContent;
    uint256 public totalSupply;
    function mintSpan(uint256 spanId, string memory content) public {
        require(ownerOf[spanId] == address(0), "Already minted");
        ownerOf[spanId] = msg.sender;
        spanContent[spanId] = content;
        totalSupply += 1;
    }
}